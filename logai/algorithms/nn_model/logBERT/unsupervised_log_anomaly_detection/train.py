#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import BertConfig, BertForMaskedLM
import math 
import os 
from .configs import LogBERTConfig
from .data_utils import load_dataset
from datasets import DatasetDict

class LogBERT:

    def __init__(self, config: LogBERTConfig):
        
        self.config = config
        self.model_name = self.config.trainer_config['model_name']
        self.tokenizer_name_or_dir = self.config.tokenizer_config['tokenizer_name_or_dir']
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_dir, use_fast=self.config.tokenizer_config['use_fast'])
        self.max_input_seq_len = self.config.tokenizer_config['max_input_seq_len']
        self.data_column_name = self.config.data_config['data_column_name']

        self.truncation = self.config.tokenizer_config['truncation']
        self.padding = self.config.tokenizer_config['padding']

        special_tokens = self.config.tokenizer_config['custom_tokens']
        special_tokens.extend([self.tokenizer.mask_token, self.tokenizer.sep_token, self.tokenizer.pad_token, self.tokenizer.unk_token, self.tokenizer.cls_token])
        ignore_tokens = [ ".", "*", ":", "$", "_", "-", "/"]
        special_tokens.extend(ignore_tokens)
        self.special_tokens = special_tokens
        
        if os.path.exists(self.config.tokenizer_config['tokenizer_name_or_dir']):
            self.custom_vocab = open(os.path.join(self.tokenizer_name_or_dir, 'vocab.txt')).readlines()

        self.pretrain_from_scratch = self.config.trainer_config['pretrain_from_scratch']
        self.mlm_probability = self.config.trainer_config['mlm_probability']

        self.model_dir = self.config.trainer_config['model_dir']

    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.data_column_name], truncation=self.truncation, padding=self.padding, max_length=self.max_input_seq_len)


    def train(self, train_file, valid_file):
        train_dataset, _, _ = load_dataset(train_file, self.data_column_name, self.special_tokens)
        valid_dataset, _, _ = load_dataset(valid_file, self.data_column_name, self.special_tokens)

        datasets = DatasetDict({"train": train_dataset, "validation": valid_dataset})

        tokenized_datasets = datasets.map(self.tokenize_function, batched=True, num_proc=4, remove_columns=[self.data_column_name])


        if os.path.exists(self.model_dir) and len(os.listdir(self.model_dir))>0:
            model_checkpoint = self.model_dir
            checkpoint_dir = 'checkpoint-'+str(max([int(x.split('-')[1]) for x in os.listdir(self.model_dir)]))
            model_checkpoint = os.path.abspath(os.path.join(model_checkpoint, checkpoint_dir))
        else:
            model_checkpoint = self.model_name


        if self.pretrain_from_scratch is False:
            model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
        else:
            vocab_size = len(self.custom_vocab)
            config = BertConfig(vocab_size=vocab_size)
            model = BertForMaskedLM(config)
            model.tokenizer = self.tokenizer


        training_args = TrainingArguments(
            self.model_dir,
            evaluation_strategy = self.config.trainer_config['evaluation_strategy'],
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

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=self.mlm_probability, pad_to_multiple_of=self.max_input_seq_len)

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



    
    

