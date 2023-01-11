#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import numpy as np
import pandas as pd
from attr import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer

from logai.algorithms.algo_interfaces import VectorizationAlgo
from logai.config_interfaces import Config
from logai.algorithms.factory import factory


@dataclass
class TfIdfParams(Config):
    """
    Configuration of TF-IDF vectorizer.
    """

    input: str = "content"
    encoding: str = "utf-8"
    decode_error: str = "strict"
    strip_accents: object = None
    lowercase: bool = True
    preprocessor: object = None
    tokenizer: object = None
    analyzer: str = "word"
    stop_words: object = None
    token_pattern: str = r"(?u)\b\w\w+\b"
    ngram_range: tuple = (1, 1)
    max_df: float = 1.0
    min_df: int = 1
    max_features: object = None
    vocabulary: object = None
    binary: bool = False
    dtype: object = np.float64
    norm: str = "l2"
    use_idf: bool = True
    smooth_idf: bool = True
    sublinear_tf: bool = False


@factory.register("vectorization", "tfidf", TfIdfParams)
class TfIdf(VectorizationAlgo):
    """
    Wrapping TF-IDF algorithm from scikit-learn.
    """

    def __init__(self, params: TfIdfParams, **kwargs):
        """
        Initialize TF-IDF vectorizer.
        :param params: TF-IDF algorithm parameters
        :param kwargs: Optional k-v based params
        """
        self.model = TfidfVectorizer(
            input=params.input,
            encoding=params.encoding,
            decode_error=params.decode_error,
            strip_accents=params.strip_accents,
            lowercase=params.lowercase,
            preprocessor=params.preprocessor,
            tokenizer=params.tokenizer,
            analyzer=params.analyzer,
            stop_words=params.stop_words,
            token_pattern=params.token_pattern,
            ngram_range=params.ngram_range,
            max_df=params.max_df,
            min_df=params.min_df,
            max_features=params.max_features,
            vocabulary=params.vocabulary,
            binary=params.binary,
            dtype=params.dtype,
            norm=params.norm,
            use_idf=params.use_idf,
            smooth_idf=params.smooth_idf,
            sublinear_tf=params.sublinear_tf,
        )

    def fit(self, loglines: pd.Series):
        """
        Train TF-IDF model.
        :param loglines: pandas.Series input training set.
        :return:
        """
        self.model.fit(loglines)
        self.vocab = self.model.vocabulary_
        self.vocab_size = len(self.vocab)

    def transform(self, loglines: pd.Series) -> pd.Series:
        """
        Transform loglines into log vectors.
        :param loglines: pandas.Series input inference set.
        :return: pandas.Series
        """
        res = self.model.transform(loglines)
        return pd.Series(res.todense().tolist(), index=loglines.index).apply(
            lambda x: np.array(x)
        )

    def summary(self):
        """
        generate model summary
        :return: TfidfVectorizer.summary
        """
        return self.model.summary()
