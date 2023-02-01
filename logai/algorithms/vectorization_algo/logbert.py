#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import os
import pandas as pd
from attr import dataclass
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from transformers import BertTokenizerFast, AutoTokenizer
from datasets import Dataset as HFDataset

from logai.algorithms.algo_interfaces import VectorizationAlgo
from logai.config_interfaces import Config
from logai.dataloader.data_model import LogRecordObject
from logai.utils import constants
from logai.algorithms.factory import factory


@dataclass
class LogBERTVectorizerParams(Config):
    """Config class for logBERT Vectorizer

    :param model_name: name of the model , using HuggingFace standardized naming.
    :param use_fast: whether to use fast tokenization or not.
    :param truncation: whether to truncate the input to max_token_len.
    :param max_token_len: maximum token length of input, if truncation is set to true.
    :param max_vocab_size: maximum size  of the vocabulary.
    :param custom_tokens: list of custom tokens.
    :param train_batch_size: batch size during training the vectorizer.
    :param output_dir: path to directory where the output would be saved.
    :param tokenizer_dirpath: path to the tokenizer where the vectorizer (logbert tokenizer) would be saved.
    :param num_proc: number of processes to be used when tokenizing.

    """

    model_name: str = ""
    use_fast: bool = True
    truncation: bool = True
    max_token_len: int = 384
    max_vocab_size: int = 5000
    custom_tokens = []
    train_batch_size: int = 1000
    output_dir: str = None
    tokenizer_dirpath: str = None
    num_proc: int = 4


@factory.register("vectorization", "logbert", LogBERTVectorizerParams)
class LogBERT(VectorizationAlgo):
    """Vectorizer class for logbert.

    :param config: A config object for specifying
        parameters of log bert vectorizer.
    """

    def __init__(self, config: LogBERTVectorizerParams):
    
        self.config = config

        self.special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]

        if self.config.custom_tokens is not None:
            self.special_tokens.extend(self.config.custom_tokens)

        tokenizer_dirname = self.config.model_name + "_tokenizer"
        if self.config.tokenizer_dirpath == "" or self.config.tokenizer_dirpath is None:
            self.config.tokenizer_dirpath = os.path.join(
                self.config.output_dir, tokenizer_dirname
            )
        if not os.path.exists(self.config.tokenizer_dirpath):
            os.makedirs(self.config.tokenizer_dirpath)

        if not os.listdir(self.config.tokenizer_dirpath):
            self.tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
            if "-uncased" in self.config.model_name:
                self.tokenizer.normalizer = normalizers.Sequence(
                    [
                        normalizers.NFD(),
                        normalizers.Lowercase(),
                        normalizers.StripAccents(),
                    ]
                )
            elif "-cased" in self.config.model_name:
                self.tokenizer.normalizer = normalizers.Sequence(
                    [normalizers.NFD(), normalizers.StripAccents()]
                )
            self.tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_dirpath, use_fast=self.config.use_fast
            )

    def fit(self, logrecord: LogRecordObject):
        """Fit method for training vectorizer for logbert.

        :param logrecord: A log record object containing the training
            dataset over which vectorizer is trained.
        """

        if os.listdir(self.config.tokenizer_dirpath):
            return

        cleaned_logrecord = self._clean_dataset(logrecord)
        dataset = self._get_hf_dataset(cleaned_logrecord)

        def batch_iterator():
            for i in range(0, len(dataset), self.config.train_batch_size):
                yield dataset[i : i + self.config.train_batch_size][
                    constants.LOGLINE_NAME
                ]

        trainer = trainers.WordPieceTrainer(
            vocab_size=self.config.max_vocab_size, special_tokens=self.special_tokens
        )

        self.tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

        cls_token_id = self.tokenizer.token_to_id("[CLS]")
        sep_token_id = self.tokenizer.token_to_id("[SEP]")

        self.tokenizer.post_processor = processors.TemplateProcessing(
            single=f"[CLS]:0 $A:0 [SEP]:0",
            pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", cls_token_id),
                ("[SEP]", sep_token_id),
            ],
        )

        self.tokenizer.decoder = decoders.WordPiece(prefix="##")
        new_tokenizer = BertTokenizerFast(tokenizer_object=self.tokenizer)

        new_tokenizer.save_pretrained(self.config.tokenizer_dirpath)
        self.tokenizer = new_tokenizer

    def _tokenize_function(self, examples):
        return self.tokenizer(
            examples[constants.LOGLINE_NAME],
            truncation=self.config.truncation,
            padding="max_length",
            max_length=self.config.max_token_len,
        )

    def _clean_dataset(self, logrecord: LogRecordObject):
        special_tokens = self._get_all_special_tokens()
        loglines = logrecord.body[constants.LOGLINE_NAME]
        loglines_removed_special_tokens = loglines.apply(
            lambda x: " ".join(set(x.split(" ")) - set(special_tokens)).strip()
        )
        indices = list(logrecord.body.loc[loglines_removed_special_tokens != ""].index)

        logrecord = logrecord.select_by_index(indices, inplace=True)
        return logrecord

    def transform(self, logrecord: LogRecordObject):
        """Transform method for running vectorizer over logrecord object.

        :param logrecord: A log record object containing the dataset
            to be vectorized.
        :return: HuggingFace dataset object.
        """
        cleaned_logrecord = self._clean_dataset(logrecord)
        dataset = self._get_hf_dataset(cleaned_logrecord)
        tokenized_dataset = dataset.map(
            self._tokenize_function,
            batched=True,
            num_proc=self.config.num_proc,
            remove_columns=[constants.LOGLINE_NAME],
        )
        return tokenized_dataset

    def _get_all_special_tokens(self):
        special_tokens = self.special_tokens
        ignore_tokens = [".", "*", ":", "$", "_", "-", "/"]
        special_tokens.extend(ignore_tokens)
        return special_tokens

    def _get_hf_dataset(self, logrecord: LogRecordObject):
        if constants.LOG_COUNTS in logrecord.attributes:
            loglines_df = pd.DataFrame(
                {
                    constants.LOGLINE_NAME: logrecord.body[constants.LOGLINE_NAME],
                    constants.LABELS: logrecord.labels[constants.LABELS],
                    constants.LOG_COUNTS: logrecord.attributes[constants.LOG_COUNTS],
                }
            )
        else:
            loglines_df = pd.DataFrame(
                {
                    constants.LOGLINE_NAME: logrecord.body[constants.LOGLINE_NAME],
                    constants.LABELS: logrecord.labels[constants.LABELS],
                }
            )
        hf_data = HFDataset.from_pandas(loglines_df)
        return hf_data
