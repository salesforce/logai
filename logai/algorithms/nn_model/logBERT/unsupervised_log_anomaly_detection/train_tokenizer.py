#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from transformers import BertTokenizerFast
import os 
from attr import dataclass
from .configs import CustomTokenizerConfig
from .data_utils import load_dataset


class CustomTokenizer:

    def __init__(self, config: CustomTokenizerConfig):
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
        

    def train_tokenizer(self, train_file, data_column_name, tokenizer_dirname):
        dataset, _, _ = load_dataset(train_file, data_column_name, self.special_tokens)
    
        def batch_iterator():
            for i in range(0, len(dataset), self.batch_size):
                yield dataset[i : i + self.batch_size][data_column_name]

        
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


