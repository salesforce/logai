#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from attr import dataclass

from logai.algorithms.nn_model.evaluate_unsup_logAD import load_dataset
from logai.config_interfaces import Config
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from transformers import BertTokenizerFast

@dataclass
class TokenizerConfig(Config):

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


class CustomTokenizer:
    def __init__(self, config: TokenizerConfig):
        """
        Initialize tokenizer with given TokenizerConfig
        :param config: TokenizerConfig
        """
        self.config = config
        model_name = self.config.tokenizer_config['model_name']
        self.tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        if "-uncased" in model_name:
            self.tokenizer.normalizer = normalizers.Sequence(
                [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])
        elif "-cased" in model_name:
            self.tokenizer.normalizer = normalizers.Sequence(
                [normalizers.NFD(), normalizers.StripAccents()])
        self.tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        self.max_vocab_size = self.config.tokenizer_config['max_vocab_size']
        self.batch_size = self.config.tokenizer_config['batch_size']
        self.special_tokens = self.config.tokenizer_config['special_tokens']
        custom_special_tokens = self.config.tokenizer_config['custom_tokens']
        if custom_special_tokens is not None:
            self.special_tokens.extend(custom_special_tokens)
        return

    def train_tokenizer(self, train_file, data_column_name, tokenizer_dirname):
        """
        Train tokenizer.

        :param train_file: the file to use for tokenizer training
        :param data_column_name: the target data columns in the train set.
        :param tokenizer_dirname: the directory to store tokenizer
        :return:
        """

        # TODO: Load datasets using data loader
        dataset, _, _ = load_dataset(train_file, data_column_name, self.special_tokens)

        def batch_iterator():
            for i in range(0, len(dataset), self.batch_size):
                yield dataset[i: i + self.batch_size][data_column_name]

        trainer = trainers.WordPieceTrainer(vocab_size=5000, special_tokens=self.special_tokens)

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
        new_tokenizer.save_pretrained(tokenizer_dirname)

        return

