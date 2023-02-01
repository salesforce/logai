#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd
from attr import dataclass

import logai.algorithms.clustering_algo
from logai.config_interfaces import Config
from logai.algorithms.factory import factory


@dataclass
class ClusteringConfig(Config):
    """Config class for Clustering algorithms.

    :param algo_name: The algorithm name.
    :param algo_params: The algorithm parameters.
    :param custom_params: Additional customized parameters.
    """
    algo_name: str = "dbscan"
    algo_params: object = None
    custom_params: object = None

    @classmethod
    def from_dict(cls, config_dict):
        config = super(ClusteringConfig, cls).from_dict(config_dict)
        config.algo_params = factory.get_config(
            "clustering", config.algo_name.lower(), config.algo_params
        )
        return config


class Clustering:
    """
    Clustering Application class defines log clustering workflow.
    It includes which algorithm to use.
    """

    def __init__(self, config: ClusteringConfig):
        self.model = factory.get_algorithm(
            "clustering", config.algo_name.lower(), config
        )

    def fit(self, log_features: pd.DataFrame):
        """Fit method of Clustering algorithm, to train on the given log features data.

        :param log_features: The training log features data.
        """
        log_features.columns = log_features.columns.astype(str)
        self.model.fit(log_features)

    def predict(self, log_features: pd.DataFrame) -> pd.Series:
        """Predict method of Clustering algorithm, to run inference on given test log features.

        :param log_features: The test log features data.
        :return: The cluster output (label).
        """
        log_features.columns = log_features.columns.astype(str)
        return self.model.predict(log_features)
