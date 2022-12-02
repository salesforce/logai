#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#

from logai.config_interfaces import Config
from attr import dataclass


@dataclass
class CustomTokenizerConfig(Config):

    tokenizer_config: dict = {
        'model_name' : "",
        'special_tokens' : ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
        'custom_tokens': [],
        'batch_size' : 1000,
        'max_vocab_size': 5000
    }

    def from_dict(self, config_dict):
        super().from_dict(config_dict)
        return 


@dataclass
class LogBERTConfig(Config):

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
