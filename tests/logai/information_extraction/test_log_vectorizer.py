#
# Copyright (c) 2023 Salesforce.com, inc.
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
from logai.algorithms.vectorization_algo.sequential import (
    SequentialVectorizerParams,
    Sequential,
)
from logai.algorithms.vectorization_algo.semantic import (
    Semantic,
    SemanticVectorizerParams,
)
from logai.algorithms.vectorization_algo.forecast_nn import (
    ForecastNNVectorizerParams,
    ForecastNN,
)
from logai.algorithms.vectorization_algo.logbert import (
    LogBERT,
    LogBERTVectorizerParams,
)
from logai.information_extraction.log_vectorizer import VectorizerConfig, LogVectorizer
from logai.dataloader.data_model import LogRecordObject
from logai.utils import constants
from tests.logai.test_utils.fixtures import (
    logrecord_body,
    hdfs_logrecord,
    bgl_logrecord,
)
from datasets import Dataset as HFDataset

TEST_OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "test_data/temp"
)


class TestLogVectorizer:
    def test_tfidf_creation(self, logrecord_body):
        params = TfIdfParams()
        config = VectorizerConfig("TFIDF", params, None)
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
        config = VectorizerConfig("Word2Vec", params, None)
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
        config = VectorizerConfig("FastText", params, None)
        vectorizer = LogVectorizer(config)
        assert isinstance(vectorizer, LogVectorizer), "not an Vectorizer object"
        assert isinstance(vectorizer.vectorizer, FastText), "not FastText model"
        loglines = logrecord_body[constants.LOGLINE_NAME]
        vectorizer.fit(loglines)
        res = vectorizer.transform(loglines)
        assert isinstance(res, pd.Series), "result is not a pandas.Series"
        assert len(loglines) == len(res), "result length should match input"
        return

    def test_sequential_creation(self, logrecord_body):
        if not os.path.exists(TEST_OUTPUT_PATH):
            os.makedirs(TEST_OUTPUT_PATH)
        params = SequentialVectorizerParams(model_save_dir=TEST_OUTPUT_PATH)
        assert params.model_save_dir == TEST_OUTPUT_PATH
        config = VectorizerConfig(
            algo_name="sequential", algo_param=params, custom_param=None
        )
        vectorizer = LogVectorizer(config)
        assert isinstance(vectorizer, LogVectorizer), "not an Vectorizer object"
        assert isinstance(
            vectorizer.vectorizer, Sequential
        ), "not Sequential vectorizer model"
        loglines = logrecord_body[constants.LOGLINE_NAME]
        vectorizer.fit(loglines)
        res = vectorizer.transform(loglines)
        assert isinstance(res, pd.Series), "result is not a pandas.Series"
        assert len(loglines) == len(res), "result length should match input"
        return

    def test_semantic_creation(self, logrecord_body):
        if not os.path.exists(TEST_OUTPUT_PATH):
            os.makedirs(TEST_OUTPUT_PATH)
        params = SemanticVectorizerParams(model_save_dir=TEST_OUTPUT_PATH)
        assert params.model_save_dir == TEST_OUTPUT_PATH
        config = VectorizerConfig("semantic", params, None)
        vectorizer = LogVectorizer(config)
        assert isinstance(vectorizer, LogVectorizer), "not an Vectorizer object"
        assert isinstance(
            vectorizer.vectorizer, Semantic
        ), "not Semantic vectorizer model"
        loglines = logrecord_body[constants.LOGLINE_NAME]
        vectorizer.fit(loglines)
        res = vectorizer.transform(loglines)
        assert isinstance(res, pd.Series), "result is not a pandas.Series"
        assert len(loglines) == len(res), "result length should match input"
        return

    def test_logbert_creation(self, bgl_logrecord):
        if not os.path.exists(TEST_OUTPUT_PATH):
            os.makedirs(TEST_OUTPUT_PATH)
        params = LogBERTVectorizerParams(
            model_name="bert-base-cased", 
            max_token_len=120, 
            output_dir=TEST_OUTPUT_PATH
        )
        assert params.model_name == "bert-base-cased"
        assert params.max_token_len == 120
        config = VectorizerConfig("logbert", params)
        vectorizer = LogVectorizer(config)
        assert isinstance(vectorizer, LogVectorizer), "not an Vectorizer object"
        assert isinstance(
            vectorizer.vectorizer, LogBERT
        ), "not Logbert vectorizer model"
        logrecord = bgl_logrecord
        vectorizer.fit(logrecord)
        res = vectorizer.transform(logrecord)
        assert isinstance(res, HFDataset), "result is not a HFDataset"
        return
