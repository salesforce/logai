#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import os

import pandas as pd

from logai.algorithms.vectorization_algo.fasttext import FastTextParams, FastText
from logai.algorithms.vectorization_algo.tfidf import TfIdfParams, TfIdf
from logai.algorithms.vectorization_algo.word2vec import Word2Vec, Word2VecParams
from logai.information_extraction.log_vectorizer import VectorizerConfig, LogVectorizer
from logai.utils import constants
from tests.logai.test_utils.fixtures import logrecord_body


class TestLogVectorizer:
    def test_tfidf_creation(self, logrecord_body):
        params = TfIdfParams()
        config = VectorizerConfig('TFIDF', params, None)
        vectorizer = LogVectorizer(config)
        assert isinstance(vectorizer, LogVectorizer), "not an Vectorizer object"
        assert isinstance(vectorizer.vectorizer, TfIdf), "not a TFIDF model"
        loglines = logrecord_body[constants.LOGLINE_NAME]
        vectorizer.fit(loglines)
        res = vectorizer.transform(loglines)
        assert isinstance(res, pd.Series), "result is not a pandas.Series"
        assert len(loglines) == len(res), "result length should match input"
        return

    def test_word2vec_creation(self, logrecord_body):
        params = Word2VecParams(max_token_len=200)
        assert params.max_token_len == 200
        config = VectorizerConfig('Word2Vec', params, None)
        vectorizer = LogVectorizer(config)
        assert isinstance(vectorizer, LogVectorizer), "not an Vectorizer object"
        assert isinstance(vectorizer.vectorizer, Word2Vec), "not a Word2Vec model"
        loglines = logrecord_body[constants.LOGLINE_NAME]
        vectorizer.fit(loglines)
        res = vectorizer.transform(loglines)
        assert isinstance(res, pd.Series), "result is not a pandas.Series"
        assert len(loglines) == len(res), "result length should match input"
        return

    def test_fasttext_creation(self, logrecord_body):
        params = FastTextParams()
        config = VectorizerConfig('FastText', params, None)
        vectorizer = LogVectorizer(config)
        assert isinstance(vectorizer, LogVectorizer), "not an Vectorizer object"
        assert isinstance(vectorizer.vectorizer, FastText), "not FastText model"
        loglines = logrecord_body[constants.LOGLINE_NAME]
        vectorizer.fit(loglines)
        res = vectorizer.transform(loglines)
        assert isinstance(res, pd.Series), "result is not a pandas.Series"
        assert len(loglines) == len(res), "result length should match input"
        return



