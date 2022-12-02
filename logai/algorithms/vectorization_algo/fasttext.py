#
# Copyright (c) 2022 Salesforce.com, inc.
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


@dataclass
class FastTextParams(Config):
    """
    Configuration for FastText vectorizer.
    """
    vector_size: int = 100
    window: int = 100
    min_count: int = 1
    sample: float = 1e-2
    workers: int = 4
    sg: int = 1
    epochs: int = 100
    max_token_len: int = 100

    def from_dict(self, config_dict):
        super().from_dict(config_dict)

        return


class FastText(VectorizationAlgo):
    """
    Wrap FastText algorithm from gensim
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
