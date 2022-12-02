#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd

import logai.utils.tokenize


def test__tokenize_camel_case():
    logline = pd.Series(["this is a camelCase logType"])

    res = logline.apply(logai.utils.tokenize._tokenize_camel_case)
    assert res[0] == "this is a camel  Case log  Type", "camelCase terms not split"


def test__tokenize_nlp():
    logline = pd.Series(["this is a camelCase logType"])
    target_tokens = [('this', 'PRON'), ('is', 'AUX'), ('a', 'DET'), ('camelCase', 'NOUN'), ('logType', 'NOUN')]
    res = logline.apply(logai.utils.tokenize._tokenize_nlp)
    assert [(w.text, w.pos_) for w in res[0]] == target_tokens, "tokens do not match"


def test__tokenize_replace_digits():
    logline = pd.Series(["this is a logline with 84392312 as digits"])
    res = logline.apply(logai.utils.tokenize._tokenize_replace_digits)
    assert "84392312" not in res[0], "digits not removed"
    assert "[DIGITS]" in res[0], "DIGITS replacement term does not exist"
