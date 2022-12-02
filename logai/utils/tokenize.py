#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
"""
Module that includes common tokenization functions to be applied by pandas dataframes.
"""

import itertools
import re
import string

import spacy as spacy
import en_core_web_sm
from transformers import AutoTokenizer

from logai.utils import constants


def replace_delimeters(logline, delimeter_regex):
    """
    Remove customer delimeters
    :param delimeter_regex:
    :param logline:
    :return:
    """
    return logline.replace(delimeter_regex, " ")


def tokenize(logline, config):
    """
    Common tokenization of logline and using space to separate tokens.
    :param config:
    :param logline:
    :return:
    """
    logline = " ".join(
        ["".join(g).strip() for k, g in itertools.groupby(logline, str.isalpha)]
    )
    logline = " ".join(
        [x for x in logline.split(" ") if len(x) > 0 and x not in string.punctuation]
    )
    return logline


def _tokenize_camel_case(logline):
    return re.sub(
        "([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", str(logline))
    ).strip()


def _tokenize_replace_digits(logline):
    tokens = []
    digits = []
    for t in logline.split():

        if t.isdigit():
            tokens.append(constants.DIGITS_SUB)
            digits.append(t)
        else:
            tokens.append(t)

    return " ".join(tokens)


def _tokenize_replace_timestamps(logline):
    pass


def _tokenize_nlp(logline):
    nlp = en_core_web_sm.load()
    return nlp(logline)
