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
class Word2VecParams(Config):
    """
    Configuration of Word2Vec vectorization parameters. For more details on the parameters see 
    https://radimrehurek.com/gensim/models/word2vec.html.

    :param max_token_len: The maximum length of tokens.
    :param min_count: Ignores all words with total frequency lower than this.
    :param vector_size: Dimensionality of the feature vectors.
    :param window: The maximum distance between the current and predicted word within a sentence.
    """

    max_token_len: int = 100
    min_count: int = 1
    vector_size: int = 3
    window: int = 3


@factory.register("vectorization", "word2vec", Word2VecParams)
class Word2Vec(VectorizationAlgo):
    """
    Word2Vec algorithm for converting raw log data into word2vec vectors. This is a wrapper class for the Word2Vec
    model from gensim library https://radimrehurek.com/gensim/models/word2vec.html

    :param max_token_len: The max token length to vectorize, longer sentences will be chopped.
    """

    def __init__(self, params: Word2VecParams):
        self.params = params
        self.model = None

    def fit(self, loglines: pd.Series):
        """
        Fits a Word2Vec model.

        :param loglines: Parsed loglines.
        """
        max_token_len = self.params.max_token_len

        doc = []
        for sentence in loglines:
            token_list = sentence.split(" ")[:max_token_len]
            for tk in token_list:
                if tk != "*":
                    doc.append(word_tokenize(tk.lower()))

        # Use Word2Vec for vectorization
        self.model = gensim.models.Word2Vec(
            doc,
            min_count=self.params.min_count,
            vector_size=self.params.vector_size,
            window=self.params.window,
        )

    def transform(self, loglines: pd.Series) -> pd.Series:
        """
        Transforms input loglines to log vectors.

        :param loglines: The input loglines.
        :return: The transformed log vectors.
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
        Generates model summary.
        """
        return self.model.summary()
