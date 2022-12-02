#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import logging
import os

import math
from transformers import TrainingArguments, Trainer

from logai.algorithms.algo_interfaces import AnomalyDetectionAlgo
from logai.algorithms.nn_model.tokenizer import Tokenizer
from logai.config_interfaces import Config

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'temp_output')

# TODO: Thinking about using algorithms.nn_model module to construct LogBERT and use this as an interface for integration


class LogBertParams(Config):
    """
    Parameters to be used in LogBERT algorithms
    TODO: get a complete list of paramters
    """

    data_config: dict = {
        'data_column_name': ''
    }

    tokenizer_config: dict = {
        'tokenizer_name_or_dir': '',
        'use_fast': True,
        'padding': 'max_length',
        'truncation': True,
        'max_input_seq_len': 384,
        'custom_tokens': []
    }

    trainer_config: dict = {
        'pretrain_from_scratch': True,
        'model_name': "bert-base-cased",
        'model_dir': '',
        'mlm_probability': 0.15,
        'evaluation_strategy': "steps",
        'num_train_epochs': 20,
        'learning_rate': 1e-5,
        'logging_steps': 10,
        'per_device_train_batch_size': 50,
        'per_device_eval_batch_size': 256,
        'weight_decay': 0.0001,
        'save_steps': 500,
        'eval_steps': 500,
        'resume_from_checkpoint': True,
    }

    eval_config: dict = {
        'per_device_eval_batch_size': 100,
        'eval_accumulation_steps': 1000,
        'mask_ngram': 1
    }

    def from_dict(self, config_dict):
        super().from_dict(config_dict)

        return


class LogBERT(AnomalyDetectionAlgo):
    def __init__(self, params:LogBertParams):
        self.tokenizer_config = params.tokenizer_config
        self.trainer_config = params.trainer_config
        self.tokenizer = None
        self.trainer = None
        return

    def tokenize(self, train_file, data_column_names):
        """
        Tokenize function create or pull an existing tokenizer and train it using the given training set.
        :param train_file:
        :param data_column_names:
        :return:
        """
        tokenizer_dir_name = self.config.tokenizer_config['tokenizer_name_or_dir']
        self.tokenizer = Tokenizer(self.config.tokenizer_config)
        self.tokenizer.train_tokenizer(
            train_file = train_file,
            data_column_names = data_column_names,
            tokenizer_dirname=tokenizer_dir_name
        )

        logging.info('Tokenizer trained and saved at: {}'.format(tokenizer_dir_name))
        return

    def train(self, train_file, valid_file):
        """
        Create a new trainer or pull existing trainer and train model using the given training data
        :param train_file:
        :param valid_file:
        :return:
        """
        # train_dataset, _, _ = load_dataset(train_file, self.data_column_name, self.special_tokens)
        # valid_dataset, _, _ = load_dataset(valid_file, self.data_column_name, self.special_tokens)
        #
        # datasets = DatasetDict({"train": train_dataset, "validation": valid_dataset})
        #
        # tokenized_datasets = datasets.map(self.tokenize_function, batched=True, num_proc=4,
        #                                   remove_columns=[self.data_column_name])
        #
        # if os.path.exists(self.model_dir) and len(os.listdir(self.model_dir)) > 0:
        #     model_checkpoint = self.model_dir
        #     checkpoint_dir = 'checkpoint-' + str(max([int(x.split('-')[1]) for x in os.listdir(self.model_dir)]))
        #     model_checkpoint = os.path.abspath(os.path.join(model_checkpoint, checkpoint_dir))
        # else:
        #     model_checkpoint = self.model_name
        #
        # if self.pretrain_from_scratch is False:
        #     model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
        # else:
        #     vocab_size = len(self.custom_vocab)
        #     config = BertConfig(vocab_size=vocab_size)
        #     model = BertForMaskedLM(config)
        #     model.tokenizer = self.tokenizer
        #
        # training_args = TrainingArguments(
        #     self.model_dir,
        #     evaluation_strategy=self.config.trainer_config['evaluation_strategy'],
        #     num_train_epochs=self.config.trainer_config['num_train_epochs'],
        #     learning_rate=self.config.trainer_config['learning_rate'],
        #     logging_steps=self.config.trainer_config['logging_steps'],
        #     per_device_train_batch_size=self.config.trainer_config['per_device_train_batch_size'],
        #     per_device_eval_batch_size=self.config.trainer_config['per_device_eval_batch_size'],
        #     weight_decay=self.config.trainer_config['weight_decay'],
        #     save_steps=self.config.trainer_config['save_steps'],
        #     eval_steps=self.config.trainer_config['eval_steps'],
        #     resume_from_checkpoint=self.config.trainer_config['resume_from_checkpoint']
        # )
        #
        # data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=self.mlm_probability,
        #                                                 pad_to_multiple_of=self.max_input_seq_len)
        #
        # trainer = Trainer(
        #     model=model,
        #     args=training_args,
        #     train_dataset=tokenized_datasets["train"],
        #     eval_dataset=tokenized_datasets["validation"],
        #     data_collator=data_collator
        # )
        #
        # trainer.train()
        #
        # eval_results = trainer.evaluate()
        # print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

        return

    def predict(self):
        return

    def _get_temp_output_path(self, path = None):

        return path if path else DEFAULT_PATH

    def _get_tokenizer(self, tokenizer_config):
        tokenizer = None
        return tokenizer

    def _get_trainer(self, trainer_config):
        training_args = TrainingArguments(
            self.model_dir,
            evaluation_strategy=self.config.trainer_config['evaluation_strategy'],
            num_train_epochs=self.config.trainer_config['num_train_epochs'],
            learning_rate=self.config.trainer_config['learning_rate'],
            logging_steps=self.config.trainer_config['logging_steps'],
            per_device_train_batch_size=self.config.trainer_config['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config.trainer_config['per_device_eval_batch_size'],
            weight_decay=self.config.trainer_config['weight_decay'],
            save_steps=self.config.trainer_config['save_steps'],
            eval_steps=self.config.trainer_config['eval_steps'],
            resume_from_checkpoint=self.config.trainer_config['resume_from_checkpoint']
        )
        #
        # trainer = Trainer(
        #     model=model,
        #     args=training_args,
        #     train_dataset=tokenized_datasets["train"],
        #     eval_dataset=tokenized_datasets["validation"],
        #     data_collator=data_collator
        # )
        return trainer


