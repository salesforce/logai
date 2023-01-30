#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from attr import dataclass
from logai.config_interfaces import Config


@dataclass
class LogBERTConfig(Config):
    """Config for logBERT model

    Inherits:
        Config: config interface

    pretrain_from_scratch: bool = True  # whether to do pretraining from scratch or intialize with the
    HuggingFace pretrained LM.
    model_name: str = "bert-base-cased" # name of the model using HuggingFace standardized naming
    model_dirname: str = None # name of the directory where the model would be saved. Directory of this name
    would be created inside `output_dir`, if it does not exist
    mlm_probability: float = 0.15 # probability of the tokens to be masked during MLM trainning
    mask_ngram: int = 1 # length of ngrams that are masked during inference
    max_token_len: int = 384 # maximum token length of the input
    learning_rate: float = 1e-5 # learning rate
    weight_decay: float = 0.0001 # parameter to use weight decay of the learning rate
    per_device_train_batch_size: int = 50 # training batch size per gpu device
    per_device_eval_batch_size: int = 256 # evaluation batch size per gpu device
    eval_accumulation_steps: int = 1000 # parameter to accumulate the evaluation results over the steps
    num_eval_shards: int = 10 # parameter to shard the evaluation data (to avoid any OOM issue)
    evaluation_strategy: str = "steps" # either steps or epoch, based on whether the unit of the eval_steps
    parameter is "steps" or "epoch"
    num_train_epochs: int = 20 # number of training epochs
    logging_steps: int = 10 # number of steps after which the output is logged
    save_steps: int = 50 # number of steps after which the model is saved
    eval_steps: int = 50 # number of steps after which evaluation is run
    resume_from_checkpoint: bool = True # whether to resume from a given model checkpoint. If set to true,
    it will find the latest checkpoint saved in the dir and use that to load the model
    output_dir: str = None # output directory where the model would be saved.
    tokenizer_dirpath: str = None # path to directory containing the tokenizer

    """

    pretrain_from_scratch: bool = True
    model_name: str = "bert-base-cased"
    model_dirname: str = None
    mlm_probability: float = 0.15
    mask_ngram: int = 1
    max_token_len: int = 384
    evaluation_strategy: str = "steps"
    num_train_epochs: int = 20
    learning_rate: float = 1e-5
    logging_steps: int = 10
    per_device_train_batch_size: int = 50
    per_device_eval_batch_size: int = 256
    eval_accumulation_steps: int = 1000
    num_eval_shards: int = 10
    weight_decay: float = 0.0001
    save_steps: int = 50
    eval_steps: int = 50
    resume_from_checkpoint: bool = True
    output_dir: str = None
    tokenizer_dirpath: str = None
