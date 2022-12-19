#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import os
from datetime import datetime

import pandas as pd
from datasets import Dataset, DatasetDict
from docutils.nodes import math
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, processors, decoders
from transformers import BertTokenizerFast, AutoTokenizer, BertConfig, BertForMaskedLM, TrainingArguments, \
    DataCollatorForLanguageModeling, Trainer

from logai.applications.application_interfaces import WorkFlowConfig
from logai.dataloader.data_loader import FileDataLoader, DataLoaderConfig
from logai.dataloader.data_model import LogRecordObject
from logai.information_extraction.feature_extractor import FeatureExtractorConfig, FeatureExtractor
from logai.information_extraction.log_parser import LogParser
from logai.preprocess.preprocessor import PreprocessorConfig, Preprocessor
from logai.utils import constants


TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "temp")


class PretrainedLMLogRepresentation:
    """
    Appliaction workflow for Pretrain LM log representation
    Support pretrain LM for given log datasets via Huggingface:
    TODO: add references. add more methods.
    """
    def __init__(self, config: WorkFlowConfig):
        self.config = config
        self.data_type = "hdfs"
        self.parse_free = False
        return

    # preprocess
    def preprocess(self, filepath, log_parser_config):
        if self.data_type == "hdfs":
            logrecord = self._process_hdfs_data(filepath)
            # if not self.parse_free:
            result = self._parse_hdfs_data(logrecord, log_parser_config)

            if not os.path.exists(TEMP_DIR):
                os.makedirs(TEMP_DIR)

            print(result.columns)
            print(os.path.join(TEMP_DIR, 'log_file_parsed.csv'))
            result.to_csv(os.path.join(TEMP_DIR, 'log_file_parsed.csv'))

        return

    # information extraction
    def tokenize(self, filepath, tokenizer_path):

        model = "-uncased"
        batch_size = 1000
        special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[BLOCK]", "[IP]", "[FILE]", "[INT]"]

        df = pd.read_csv(filepath, header=0)

        logline = df[[constants.LOGLINE_NAME]]

        tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        if "-uncased" in model:
            tokenizer.normalizer = normalizers.Sequence(
                [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])
        elif "-cased" in model:
            tokenizer.normalizer = normalizers.Sequence(
                [normalizers.NFD(), normalizers.StripAccents()])
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        def batch_iterator():
            for i in range(0, len(logline), batch_size):
                yield logline[i: i + batch_size][constants.LOGLINE_NAME]

        trainer = trainers.WordPieceTrainer(vocab_size=5000, special_tokens=special_tokens)

        tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

        cls_token_id = tokenizer.token_to_id("[CLS]")
        sep_token_id = tokenizer.token_to_id("[SEP]")

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"[CLS]:0 $A:0 [SEP]:0",
            pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", cls_token_id),
                ("[SEP]", sep_token_id),
            ],
        )

        tokenizer.decoder = decoders.WordPiece(prefix="##")
        new_tokenizer = BertTokenizerFast(tokenizer_object=tokenizer)
        new_tokenizer.save_pretrained(tokenizer_path)
        return

    def featurize(self, logrecord, config, labelpath):

        dir = os.path.join(TEMP_DIR, "feature_df.csv")

        data = 'hdfs'
        has_label = True
        if data == "hdfs":
            feature_df = self._featurize_hdfs_data(logrecord, config, labelpath)

        feature_df.to_csv(dir)

        self._train_test_split(dir, test_size=0.22)

        return

    def train(self, train_file_path, val_file_path, tokenizer_dir):
        model_name = "pretrained_lm_model"
        model_dir = os.path.join(TEMP_DIR, model_name)

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
        pass

    def predict(self):
        pass

    def _process_hdfs_data(self, filepath):
        file_config = DataLoaderConfig(
            filepath=filepath,
            log_type="log",
            dimensions={
                "timestamp": ["Date", "Time"],
                "body": ["Content"],
                "attributes": ["Level"],
                "span_id": ["Pid"],
            },
            reader_args={
                "log_format": "<Date> <Time> <Pid> <Level> <Content>",
            },
        )

        dataloader = FileDataLoader(file_config)
        logrecord = dataloader.load_data()
        date_time_format = '%y%m%d %H%M%S'
        logrecord.timestamp[constants.LOG_TIMESTAMPS] = (logrecord.timestamp[constants.LOG_TIMESTAMPS]).apply(
            lambda x: datetime.strptime(x, date_time_format))

        preprocessor_config = PreprocessorConfig(
            custom_replace_list=[
                [r"(?<=blk_)[-\d]+", "<block_id>"],
                [r"\d+\.\d+\.\d+\.\d+", "<IP>"],
                [r"(/[-\w]+)+", "<file_path>"],
            ]
        )

        loglines = logrecord.body[constants.LOGLINE_NAME]

        preprocessor = Preprocessor(preprocessor_config)

        clean_logs, custom_patterns = preprocessor.clean_log(loglines)

        logrecord.attributes['block_id'] = custom_patterns['<block_id>']
        logrecord.attributes['block_id'] = logrecord.attributes['block_id'].apply(lambda x: ', '.join(set(x)))
        logrecord.body[constants.LOGLINE_NAME] = clean_logs

        return logrecord

    def _parse_hdfs_data(self, logrecord, config):
        loglines = logrecord.body[constants.LOGLINE_NAME]
        parser = LogParser(config)
        parsed_result = parser.parse(loglines)

        parsed_result = parsed_result.join(logrecord.timestamp[constants.LOG_TIMESTAMPS]).join(logrecord.attributes)
        return parsed_result

    def _featurize_hdfs_data(self, logrecord: LogRecordObject, config: FeatureExtractorConfig, labelpath=None):
        feature_extractor = FeatureExtractor(config)
        event_ids, event_sequence = feature_extractor.convert_to_sequence(
            log_pattern=logrecord.body[constants.LOGLINE_NAME],
            attributes=logrecord.attributes,
            timestamps=logrecord.timestamp[constants.LOG_TIMESTAMPS])

        event_seq_df = event_ids[['block_id']].join(event_sequence.rename('event_sequence'))

        if labelpath:
            blk_df = pd.read_csv(labelpath, header=0)
            anomaly_blk = set(blk_df[blk_df['Label'] == 'Anomaly']['BlockId'])

            event_seq_df['label'] = event_seq_df['block_id'].apply(lambda x: 1 if x in anomaly_blk else 0)
            event_seq_df['event_id'] = event_seq_df['block_id']
        return event_seq_df[['event_id', 'event_sequence', 'label']]

    def _train_test_split(self, filepath, test_size):

        dir = os.path.dirname(filepath)

        name = filepath.split('/')[-1].split('.')[0]

        feature_df = pd.read_csv(filepath, header=0)

        train_df, test_df = train_test_split(feature_df, test_size=test_size)

        train_df.to_csv(os.path.join(dir, name + "_train.csv"))
        test_df.to_csv(os.path.join(dir, name + "_test.csv"))
        return






