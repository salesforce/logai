#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd
from attr import dataclass

import logai.algorithms.vectorization_algo
from logai.config_interfaces import Config
from logai.algorithms.factory import factory


@dataclass
class VectorizerConfig(Config):
    algo_name: str = "word2vec"
    algo_param: object = None
    custom_param: object = None

    def from_dict(self, config_dict):
        super().from_dict(config_dict)
        self.algo_param = factory.get_config(
            "vectorization", self.algo_name.lower(), self.algo_param)


class LogVectorizer:
    """
    Implement Log Vectorizer. Support Word2Vec and FastText vectorization.
    """

    def __init__(self, config: VectorizerConfig):
        name = config.algo_name.lower()
        config_class = factory.get_config_class("vectorization", name)
        algorithm_class = factory.get_algorithm_class("vectorization", name)
        self.vectorizer = algorithm_class(
            config.algo_param if config.algo_param else config_class())

    def fit(self, loglines: pd.Series):
        self.vectorizer.fit(loglines)

    def transform(self, loglines: pd.Series) -> pd.Series:
        return self.vectorizer.transform(loglines)
