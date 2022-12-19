#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd
from attr import dataclass
from logai.algorithms.vectorization_algo.semantic import (
    Semantic,
    SemanticVectorizerParams,
)
from logai.algorithms.vectorization_algo.sequential import (
    Sequential,
    SequentialVectorizerParams,
)
from logai.algorithms.vectorization_algo.logbert import (
    LogBERT,
    LogBERTVectorizerParams,
)
from logai.algorithms.vectorization_algo.forecast_nn import (
    ForecastNN,
    ForecastNNVectorizerParams,
)
from logai.algorithms.vectorization_algo.fasttext import FastText, FastTextParams
from logai.algorithms.vectorization_algo.tfidf import TfIdf, TfIdfParams
from logai.algorithms.vectorization_algo.word2vec import Word2Vec, Word2VecParams
from logai.config_interfaces import Config


@dataclass
class VectorizerConfig(Config):
    algo_name: str = "word2vec"
    algo_param: object = None
    custom_param: object = None

    def from_dict(self, config_dict):
        super().from_dict(config_dict)
        if self.algo_name:
            if self.algo_name.lower() == "word2vec":
                algo_param = Word2VecParams()
                algo_param.from_dict(self.algo_param)
                self.algo_param = algo_param

            elif self.algo_name.lower() == "tfidf":
                algo_param = TfIdfParams()
                algo_param.from_dict(self.algo_param)
                self.algo_param = algo_param

            elif self.algo_name.lower() == "fasttext":
                algo_param = FastTextParams()
                algo_param.from_dict(self.algo_param)
                self.algo_param = algo_param

            elif self.algo_name.lower() == "semantic":
                algo_param = SemanticVectorizerParams()
                algo_param.from_dict(self.algo_param)
                self.algo_param = algo_param

            elif self.algo_name.lower() == "sequential":
                algo_param = SequentialVectorizerParams()
                algo_param.from_dict(self.algo_param)
                self.algo_param = algo_param

            elif self.algo_name.lower() == "logbert":
                algo_param = LogBERTVectorizerParams()
                algo_param.from_dict(self.algo_param)
                self.algo_param = algo_param

            elif self.algo_name.lower() == "forecast_nn":
                algo_param = ForecastNNVectorizerParams()
                algo_param.from_dict(self.algo_param)
                self.algo_param = algo_param
        return


class LogVectorizer:
    """
    Implement Log Vectorizer. Support Word2Vec and FastText vectorization.
    """

    def __init__(self, config: VectorizerConfig):
        if config.algo_name.lower() == "word2vec":
            self.vectorizer = Word2Vec(
                config.algo_param if config.algo_param else Word2VecParams()
            )
        elif config.algo_name.lower() == "tfidf":
            self.vectorizer = TfIdf(
                config.algo_param if config.algo_param else TfIdfParams()
            )
        elif config.algo_name.lower() == "fasttext":
            self.vectorizer = FastText(
                config.algo_param if config.algo_param else FastTextParams()
            )
        elif config.algo_name.lower() == "semantic":
            self.vectorizer = Semantic(
                config.algo_param if config.algo_param else SemanticVectorizerParams()
            )
        elif config.algo_name.lower() == "sequential":
            self.vectorizer = Sequential(
                config.algo_param if config.algo_param else SequentialVectorizerParams()
            )
        elif config.algo_name.lower() == "logbert":
            self.vectorizer = LogBERT(
                config.algo_param if config.algo_param else LogBERTVectorizerParams()
            )
        elif config.algo_name.lower() == "forecast_nn":
            self.vectorizer = ForecastNN(
                config.algo_param if config.algo_param else ForecastNNVectorizerParams()
            )
        else:
            raise RuntimeError("Vectorizer {} is not defined".format(config.algo_name))
        return

    def fit(self, loglines: pd.Series):
        self.vectorizer.fit(loglines)
        return

    def transform(self, loglines: pd.Series) -> pd.Series:
        return self.vectorizer.transform(loglines)
