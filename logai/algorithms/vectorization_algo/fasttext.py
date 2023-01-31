#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import gensim
import numpy as np
import pandas as pd
from attr import dataclass

from nltk.tokenize import word_tokenize
from logai.algorithms.algo_interfaces import VectorizationAlgo
from logai.config_interfaces import Config
from logai.algorithms.factory import factory


@dataclass
class FastTextParams(Config):
    """
    Configuration for FastText vectorizer. For more details on the parameters see
    https://radimrehurek.com/gensim/models/fasttext.html

    param: vector_size: int = 100
            size of vector
    param: window: int = 100: optional
            The maximum distance between the current and predicted word within a sentence.
    param: min_count: int = 1: optional
            Ignores all words with total frequency lower than this.
    param: sample: float = 1e-2: optional
            The threshold for configuring which higher-frequency words are randomly downsampled,
            useful range is (0, 1e-5).
    param: workers: int = 4: number of workers to run
    param: sg: int = 1: optional. Training algorithm: skip-gram if `sg=1`, otherwise CBOW.
    param: epochs: int = 100.
    param: max_token_len: int = 100  : max token length.

    """

    vector_size: int = 100
    window: int = 100
    min_count: int = 1
    sample: float = 1e-2
    workers: int = 4
    sg: int = 1
    epochs: int = 100
    max_token_len: int = 100


@factory.register("vectorization", "fasttext", FastTextParams)
class FastText(VectorizationAlgo):
    """
    This is a wrapper for FastText algorithm from gensim library. For details see
    https://radimrehurek.com/gensim/models/fasttext.html
    """

    def __init__(self, params: FastTextParams):
        """
        Initialize FastText vectorizer
        :param max_token_len: the max token length to vectorize, longer sentences will be chopped.
        """
        self.params = params
        self.model = None

    def fit(self, loglines: pd.Series):
        """
        fit FastText model.
        :param loglines: parsed loglines.
        :return:
        """
        max_token_len = self.params.max_token_len

        doc = []
        for sentence in loglines:
            token_list = sentence.split(" ")[:max_token_len]
            for tk in token_list:
                if tk != "*":
                    doc.append(word_tokenize(tk.lower()))

        # Use Word2Vec for vectorization
        self.model = gensim.models.FastText(
            doc,
            vector_size=self.params.vector_size,
            window=self.params.window,
            min_count=self.params.min_count,
            sample=self.params.sample,
            workers=self.params.workers,
            sg=self.params.sg,
            epochs=self.params.epochs,
        )

    def transform(self, loglines: pd.Series) -> pd.Series:
        """
        Transform input loglines to log vectors
        :param loglines: pd.Series: input loglines
        :return: pd.Series
        """
        log_vectors = []
        max_len = 0
        for ll in loglines:
            token_list = ll.split(" ")

            log_vector = []

            token_list = token_list[: self.params.max_token_len]

            max_len = max(max_len, len(token_list))
            for tk in token_list:
                if tk == "*":
                    continue
                log_vector.append(self.model.wv[word_tokenize(tk.lower())][0])
            log_vectors.append(np.array(log_vector).flatten())
        log_vector_series = pd.Series(log_vectors, index=loglines.index)
        return log_vector_series

    def summary(self):
        """
        generate model summary
        :return:
        """
        return self.model.summary()
