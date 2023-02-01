#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import numpy as np
import pandas as pd
import torch
from typing import Tuple, Dict
from attr import dataclass
from datasets import Dataset, load_metric
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
)

from logai.config_interfaces import Config
from logai.utils import constants


@dataclass
class TransformerAlgoConfig(Config):
    """Config class for Transformer based model for log classification tasks.
    """
    tokenizer_config: dict = {"name": "auto", "model": "bert-base-cased"}
    trainer_config: dict = {}


class LogDataset(torch.utils.data.Dataset):
    """Wrapper class for Log Dataset, to wrap over torch Dataset class.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class TransformerAlgo:
    """HuggingFace Transformer based Pretrained Language model (e.g. "bert-base-cased"), 
    with a sequence classifier head for any supervised log classification task.
    For e.g. log anomaly detection is one type of log classfication task where the labels
    are Normal (Label 0) or Anomalous (Label 1). Currently it supports only binary 
    classification, to change this `num_labels` of AutoModelForSequenceClassification 
    has to be changed accordingly along with the prediction logic in predict method.
    """
    def __init__(self, config: TransformerAlgoConfig):
        self.config = config
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.tokenizer_config["model"], num_labels=2
        )
        self.trainer = None
        self.tokenizer = None
        self.metric = load_metric("accuracy")

        return

    def save(self, output_dir: str):
        """Save model in given directory.

        :param output_dir: The path to output directory where model should be dumped.
        """
        self.trainer.save_model(output_dir)
        return

    def train(self, train_logs: pd.Series, train_labels: pd.Series):
        """Train method for Transformer based pretrained language model with
        a sequence classification head for supervised log classification task. 
        Internally this method also splits the available training logs into train and dev data.

        :param train_logs: The training log vectors data (after LogVectorizer).
        :param train_labels: The training label data.
        """
        train_logs = train_logs.rename(constants.LOG_EVENTS)
        if not self.tokenizer:
            if self.config.tokenizer_config["name"] == "auto":
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.tokenizer_config["model"]
                )
            else:
                raise ValueError("Tokenizer is not supported.")

        train_logs, val_logs, train_labels, val_labels = train_test_split(
            train_logs, train_labels, test_size=0.2
        )

        train_df = pd.concat((train_logs, train_labels), axis=1)
        val_df = pd.concat((val_logs, val_labels), axis=1)

        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)

        tokenized_train_datasets = train_dataset.map(
            self._tokenize_auto_tokenizer, batched=True
        )
        tokenized_val_datasets = val_dataset.map(
            self._tokenize_auto_tokenizer, batched=True
        )

        training_args = TrainingArguments(
            output_dir="test_trainer",
            evaluation_strategy="epoch",
            num_train_epochs=2,
            learning_rate=1e-4,
            max_steps=10,
            do_eval=False,
            do_predict=False,
            per_device_train_batch_size=5,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train_datasets,
            eval_dataset=tokenized_val_datasets,
            # compute_metrics=self._compute_metrics,
        )

        self.trainer.train()

    def train_with_native_torch(self, train_logs: pd.Series, train_labels: pd.Series):
        """
        Train models in native torch way.

        :param train_logs: The training log features data (after LogVectorizer).
        :param train_labels: The label data for training logs.
        """
        if not self.tokenizer:
            if self.config.tokenizer_config["name"] == "auto":
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.tokenizer_config["model"]
                )
            else:
                raise ValueError("Tokenizer is not supported.")

        train_logs, val_logs, train_labels, val_labels = train_test_split(
            train_logs, train_labels, test_size=0.2
        )

        train_encodings = self.tokenizer(
            train_logs.tolist(), truncation=True, padding=True
        )

        train_dataset = LogDataset(train_encodings, train_labels.tolist())

        self._train(train_dataset=train_dataset)
        self.model.eval()
        return

    def predict(self, test_logs: pd.Series, test_labels: pd.Series) -> Tuple[pd.Series, np.ndarray, Dict[str, float]]:
        """Predict method for running evaluation on test log data.

        :param test_logs: The test log features data (output of LogVectorizer).
        :param test_labels: The labels of test log data.
        :return: - res (pd.Series): Predicted test labels as pandas Series object.
            - label_ids (`np.ndarray`, *optional*): True test labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics.

        """
        test_logs = test_logs.rename(constants.LOG_EVENTS)
        if not self.tokenizer:
            if self.config.tokenizer_config["name"] == "auto":
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.tokenizer_config["model"]
                )
            else:
                raise ValueError("Tokenizer is not supported.")

        if not test_labels.empty:
            test_df = pd.concat((test_logs, test_labels), axis=1)
        else:
            test_df = pd.DataFrame(test_logs)
        test_dataset = Dataset.from_pandas(test_df)

        tokenized_test_datasets = test_dataset.map(
            self._tokenize_auto_tokenizer, batched=True
        )

        if self.trainer:
            predictions, label_ids, metrics = self.trainer.predict(
                tokenized_test_datasets
            )

            res = pd.Series(
                [0 if p[0] > p[1] else 1 for p in predictions], test_logs.index
            )
            return res, label_ids, metrics
        else:
            raise RuntimeError("No trainer found.")

    def _tokenize_auto_tokenizer(self, examples):
        return self.tokenizer(
            examples[constants.LOG_EVENTS], padding="max_length", truncation=True
        )

    def _compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)

    def _train(self, train_dataset):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.tokenizer_config["model"]
        )
        model.to(device)
        model.train()

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        optim = AdamW(model.parameters(), lr=5e-5)

        for epoch in range(3):
            for batch in train_loader:
                optim.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                loss.backward()
                optim.step()
