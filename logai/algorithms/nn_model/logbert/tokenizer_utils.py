#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from transformers import AutoTokenizer
import os


def get_tokenizer(tokenizer_dirpath):
    """Get huggingface tokenizer object from a given directory path.

    :param tokenizer_dirpath: (str): absolute path to directory containing pretrained tokenizer.
    :return:  AutoTokenizer: tokenizer object.
    """
    return AutoTokenizer.from_pretrained(tokenizer_dirpath, use_fast=True)


def get_special_tokens():
    """gets special tokens

    :return: list of special tokens
    """
    return [
        "[UNK]",
        "[PAD]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        ".",
        "*",
        ":",
        "$",
        "_",
        "-",
        "/",
    ]


def get_special_token_ids(tokenizer):
    """Get ids of special tokens, given a tokenizer object.

    :param tokenizer: (AutoTokenizer): tokenizer object.
    :return: list of token ids of special tokens.
    """
    return [tokenizer.convert_tokens_to_ids(x) for x in get_special_tokens()]


def get_tokenizer_vocab(tokenizer_dirpath):
    """Get vocabulary from a given tokenizer directory path.

    :param tokenizer_dirpath: (str): absolute path to directory containing pretrained tokenizer.
    :return: list of vocabulary words.
    """
    return open(os.path.join(tokenizer_dirpath, "vocab.txt")).readlines()


def get_mask_id(tokenizer):
    """Get id of mask token, given a tokenizer object.
    
    :param tokenizer: (AutoTokenizer): tokenizer object.
    :return: id of mask token.
    """
    return tokenizer.convert_tokens_to_ids("[MASK]")
