#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import BertConfig, BertForMaskedLM
from datasets import Dataset as HFDataset
import math
import os
from .configs import LogBERTConfig
from .tokenizer_utils import get_tokenizer, get_tokenizer_vocab
import logging


class LogBERTTrain:
    """Class for training logBERT model to learn log representations"""

    def __init__(self, config: LogBERTConfig):

        self.config = config

        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)

        self.pretrain_from_scratch = self.config.pretrain_from_scratch
        self.mlm_probability = self.config.mlm_probability

        self.model_dirpath = os.path.join(
            self.config.output_dir, self.config.model_name
        )
        if not os.path.exists(self.model_dirpath):
            os.makedirs(self.model_dirpath)

        self.tokenizer = get_tokenizer(self.config.tokenizer_dirpath)
        self.custom_vocab = get_tokenizer_vocab(self.config.tokenizer_dirpath)

    def _initialize_trainer(self, model, train_dataset, dev_dataset):
        """initializing huggingface trainer object for logbert"""
        training_args = TrainingArguments(
            self.model_dirpath,
            evaluation_strategy=self.config.evaluation_strategy,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            logging_steps=self.config.logging_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            weight_decay=self.config.weight_decay,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            resume_from_checkpoint=self.config.resume_from_checkpoint,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm_probability=self.config.mlm_probability,
            pad_to_multiple_of=self.config.max_token_len,
        )

        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=data_collator,
        )

    def get_model_checkpoint(self):
        """Get the latest dumped checkpoint from the model directory path mentioned in logBERTConfig.

        :return: path to model checkpoint (or name of model in case of a pretrained model from hugging face).
        """
        if os.path.exists(self.model_dirpath) and os.listdir(self.model_dirpath):
            checkpoint_dir = "checkpoint-" + str(
                max(
                    [
                        int(x.split("-")[1])
                        for x in os.listdir(self.model_dirpath)
                        if x.startswith("checkpoint-")
                    ]
                )
            )
            model_checkpoint = os.path.abspath(
                os.path.join(self.model_dirpath, checkpoint_dir)
            )
        else:
            model_checkpoint = self.config.model_name
        return model_checkpoint

    def fit(self, train_dataset: HFDataset, dev_dataset: HFDataset):
        """Fit method for training logbert model.

        :param train_dataset: training dataset of type huggingface Dataset object.
        :param dev_dataset: development dataset of type huggingface Dataset object.
        """
        model_checkpoint = self.get_model_checkpoint()

        if self.pretrain_from_scratch is False:
            model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
        else:
            vocab_size = len(self.custom_vocab)
            config = BertConfig(vocab_size=vocab_size)
            model = BertForMaskedLM(config)
            model.tokenizer = self.tokenizer

        self._initialize_trainer(model, train_dataset, dev_dataset)

        self.trainer.train()

    def evaluate(self):
        """Evaluate methof for evaluating logbert model on dev data using perplexity metric."""
        eval_results = self.trainer.evaluate()
        logging.info("Perplexity: {}".format(math.exp(eval_results["eval_loss"])))
