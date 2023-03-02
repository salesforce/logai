#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#

from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer


class TestTokenizer:
    def setup(self):
        pass

    def tokenize(self):

        tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        if "-uncased" in model:
            tokenizer.normalizer = normalizers.Sequence(
                [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])
        elif "-cased" in model:
            tokenizer.normalizer = normalizers.Sequence(
                [normalizers.NFD(), normalizers.StripAccents()])
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
