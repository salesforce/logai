#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from transformers import AutoModelForMaskedLM, BertForSequenceClassification, BertTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import BertConfig, BertForMaskedLM
from datasets import Dataset as HFDataset
import math
import os
from .configs import LogBERTConfig
from .tokenizer_utils import get_tokenizer, get_tokenizer_vocab
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


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
            do_train=True,
            do_eval=True,
            warmup_steps=100,
            logging_dir=self.model_dirpath,
            evaluation_strategy=self.config.evaluation_strategy,
            fp16=True,
            load_best_model_at_end=True,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            logging_steps=self.config.logging_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            weight_decay=self.config.weight_decay,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            save_steps=self.config.save_steps,
            save_strategy=self.config.save_strategy,
            save_total_limit=self.config.save_total_limit,
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

    def fit(self, train_dataset: HFDataset, dev_dataset: HFDataset, **kwargs):
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

        if self.config.resume_from_checkpoint:
            self.trainer.train(resume_from_checkpoint=self.config.resume_from_checkpoint)
        else:
            self.trainer.train()

    def evaluate(self):
        """Evaluate methof for evaluating logbert model on dev data using perplexity metric."""
        eval_results = self.trainer.evaluate()
        logging.info("Perplexity: {}".format(math.exp(eval_results["eval_loss"])))

class LogBERTMultiClassificationTrain:

    """Class for training logBERT model to learn log representations"""

    def __init__(self, config: LogBERTConfig):

        self.model = None
        self.config = config

        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)

        self.pretrain_from_scratch = self.config.pretrain_from_scratch

        self.model_dirpath = os.path.join(
            self.config.output_dir, self.config.model_name
        )
        if not os.path.exists(self.model_dirpath):
            os.makedirs(self.model_dirpath)

    def _initialize_trainer(self, model, train_dataset, dev_dataset):
        """initializing huggingface trainer object for logbert"""
        training_args = TrainingArguments(
            output_dir=self.model_dirpath,
            do_train=True,
            do_eval=True,
            #  The number of epochs, defaults to 3.0
            num_train_epochs = self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            # Number of steps used for a linear warmup
            warmup_steps=100,
            weight_decay=self.config.weight_decay,
            logging_strategy='steps',
            # TensorBoard log directory
            logging_dir='./multi-class-logs',
            logging_steps=self.config.logging_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            #fp16=True,
            load_best_model_at_end=True,
            ## different params
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            learning_rate=self.config.learning_rate,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            resume_from_checkpoint=self.config.resume_from_checkpoint,
        )

        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=self.compute_metrics
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

    def fit(self, train_dataset: HFDataset, dev_dataset: HFDataset, **kwargs):
        """Fit method for training logbert model.

        :param train_dataset: training dataset of type huggingface Dataset object.
        :param dev_dataset: development dataset of type huggingface Dataset object.
        """
        model_checkpoint = self.get_model_checkpoint()

        self.model = BertForSequenceClassification.from_pretrained(model_checkpoint,
                                                              num_labels=kwargs.get("num_labels"),
                                                              id2label=kwargs.get("id2label"),
                                                              label2id=kwargs.get("label2id"))
        self.model.to(kwargs.get("device"))

        self._initialize_trainer(self.model, train_dataset, dev_dataset)

        if self.config.resume_from_checkpoint:
            self.trainer.train(resume_from_checkpoint=self.config.resume_from_checkpoint)
        else:
            self.trainer.train()

    def evaluate(self):
        """Evaluate methof for evaluating logbert model on dev data using perplexity metric."""
        eval_results = self.trainer.evaluate()
        logging.info("Perplexity: {}".format(math.exp(eval_results["eval_loss"])))

    def compute_metrics(self, pred):
        """
        Computes accuracy, F1, precision, and recall for a given set of predictions.

        Args:
            pred (obj): An object containing label_ids and predictions attributes.
                - label_ids (array-like): A 1D array of true class labels.
                - predictions (array-like): A 2D array where each row represents
                  an observation, and each column represents the probability of
                  that observation belonging to a certain class.

        Returns:
            dict: A dictionary containing the following metrics:
                - Accuracy (float): The proportion of correctly classified instances.
                - F1 (float): The macro F1 score, which is the harmonic mean of precision
                  and recall. Macro averaging calculates the metric independently for
                  each class and then takes the average.
                - Precision (float): The macro precision, which is the number of true
                  positives divided by the sum of true positives and false positives.
                - Recall (float): The macro recall, which is the number of true positives
                  divided by the sum of true positives and false negatives.
        """
        # Extract true labels from the input object
        labels = pred.label_ids

        # Obtain predicted class labels by finding the column index with the maximum probability
        preds = pred.predictions.argmax(-1)

        # Compute macro precision, recall, and F1 score using sklearn's precision_recall_fscore_support function
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')

        # Calculate the accuracy score using sklearn's accuracy_score function
        acc = accuracy_score(labels, preds)

        # Return the computed metrics as a dictionary
        return {
            'Accuracy': acc,
            'F1': f1,
            'Precision': precision,
            'Recall': recall
        }



