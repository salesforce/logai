#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import os.path
from datetime import datetime

import math
import pandas as pd
import pytest
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, processors, decoders
from transformers import BertTokenizerFast, AutoTokenizer, TrainingArguments, BertConfig, BertForMaskedLM, \
    DataCollatorForLanguageModeling, Trainer

from logai.applications.application_interfaces import WorkFlowConfig
from logai.applications.pretrained_lm_log_representation import PretrainedLMLogRepresentation
from logai.dataloader.data_loader import DataLoaderConfig, FileDataLoader
from logai.information_extraction.categorical_encoder import CategoricalEncoderConfig, CategoricalEncoder
from logai.information_extraction.feature_extractor import FeatureExtractorConfig, FeatureExtractor
from logai.information_extraction.log_parser import LogParser, LogParserConfig
from logai.preprocess.preprocess import PreprocessorConfig, Preprocessor
from logai.utils import constants

TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "temp")

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")


class TestPretrainedLMLogRepresentation:
    def setup(self):

        print(TEMP_DIR)

        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)
        pass

    def test_preprocess(self):

        filepath = os.path.join(TEST_DATA_DIR, "HDFS/HDFS_2000.log")
        log_parse_config = LogParserConfig()
        config = WorkFlowConfig()
        app = PretrainedLMLogRepresentation(config)
        app.preprocess(filepath, log_parse_config)
        return

    # def test_aget_file(self):
    #     filepath = os.path.join(TEST_DATA_DIR, "HDFS/HDFS.log")
    #     to_file = os.path.join(TEST_DATA_DIR, "HDFS/HDFS_2000.log")
    #     with open(to_file, "w") as wf:
    #         with open(filepath, "r") as f:
    #             lines = f.readlines()
    #             count = 0
    #             for l in lines:
    #                 if count > 2000:
    #                     return
    #                 wf.write(l)
    #                 count += 1


    def test__preprocess_hdfs_data(self):
        filepath = os.path.join(TEST_DATA_DIR, "HDFS/HDFS_2000.log")
        config = WorkFlowConfig()
        app = PretrainedLMLogRepresentation(config)

        logrecord = app._process_hdfs_data(filepath)

        assert constants.LOG_TIMESTAMPS in logrecord.timestamp.columns, "Timestamp should be exist"
        assert "block_id" in logrecord.attributes.columns, "block_id should be in attributes"


        return

    def test_tokenize(self):
        PARSED_LOG_DIR = os.path.join(TEMP_DIR, 'log_file_parsed.csv')
        TOKENIZER_DIR = os.path.join(TEMP_DIR, 'tokenizer_file')
        config = WorkFlowConfig()
        app = PretrainedLMLogRepresentation(config)

        app.tokenize(PARSED_LOG_DIR, TOKENIZER_DIR)

        return

    def test__featurize_hdfs_data(self):
        filepath = os.path.join(TEST_DATA_DIR, "HDFS/HDFS_2000.log")
        labelpath = os.path.join(TEST_DATA_DIR, "HDFS/anomaly_label.csv")
        config = WorkFlowConfig()
        app = PretrainedLMLogRepresentation(config)

        logrecord = app._process_hdfs_data(filepath)

        featurizer_config = FeatureExtractorConfig(group_by_category=['block_id'], group_by_time='1min')

        feature_df = app._featurize_hdfs_data(logrecord, featurizer_config, labelpath)

        return

    def test_featurize(self):
        filepath = os.path.join(TEST_DATA_DIR, "HDFS/HDFS_2000.log")
        labelpath = os.path.join(TEST_DATA_DIR, "HDFS/anomaly_label.csv")
        config = WorkFlowConfig()
        app = PretrainedLMLogRepresentation(config)
        logrecord = app._process_hdfs_data(filepath)

        featurizer_config = FeatureExtractorConfig(group_by_category=['block_id'], group_by_time='1min')

        app.featurize(logrecord, featurizer_config, labelpath)

        return

    @pytest.mark.skip(reason="too time consuming")
    def test_train(self):

        model_name = 'unilog'

        model_dir = os.path.join(TEMP_DIR, 'model_name')

        train_file_path = os.path.join(TEMP_DIR, 'feature_df_train.csv')
        val_file_path = os.path.join(TEMP_DIR, 'feature_df_test.csv')

        tokenizer_dir = os.path.join(TEMP_DIR, 'tokenizer_file')

        train_df = pd.read_csv(train_file_path, header=0)

        val_df = pd.read_csv(val_file_path, header=0)

        train_set = Dataset.from_pandas(train_df[['event_sequence']])
        val_set = Dataset.from_pandas(val_df[['event_sequence']])

        datasets = DatasetDict({
            "train": train_set,
            "validation": val_set
        })

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True, local_files_only=True)

        def tokenize_function(examples):
            return tokenizer(
                examples['event_sequence'],
                truncation=True,
                padding='max_length',
                max_length=300
            )

        tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=['event_sequence'])

        print(tokenized_datasets)

        if os.path.exists(model_dir) and len(os.listdir(model_dir))>0:
            model_checkpoint = model_dir
            # TODO: versioning
            checkpoint_dir = 'checkpoint-'+str(max([int(x.split('-')[1]) for x in os.listdir(model_dir)]))
            model_checkpoint = os.path.abspath(os.path.join(model_checkpoint, checkpoint_dir))
        else:
            model_checkpoint = model_name

        custom_vocab = open(os.path.join(tokenizer_dir, 'vocab.txt')).readlines()
        vocab_size = len(custom_vocab)
        config = BertConfig(vocab_size=vocab_size)
        model = BertForMaskedLM(config)
        model.tokenizer = tokenizer
        training_args = TrainingArguments(
            model_dir,
            evaluation_strategy='epoch',
            num_train_epochs=100,
            learning_rate=0.01,
            per_device_train_batch_size=10,
            per_device_eval_batch_size=10,
            weight_decay=0.1,
            save_steps=10,
            eval_steps=10,
            resume_from_checkpoint=model_dir
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=0.1,
            pad_to_multiple_of=100)


        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator
        )

        trainer.train()

        eval_results = trainer.evaluate()
        print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

        return


















