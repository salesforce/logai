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
from sklearn.cluster import Birch

from logai.algorithms.algo_interfaces import ClusteringAlgo
from logai.config_interfaces import Config
from logai.algorithms.factory import factory


@dataclass
class BirchParams(Config):
    branching_factor: int = 50
    n_clusters: int = None
    threshold: float = 1.5

    def from_dict(self, config_dict):
        super().from_dict(config_dict)


@factory.register("clustering", "birch", BirchParams)
class BirchAlgo(ClusteringAlgo):
    """
    Implement BIRCH for log clustering.
    """

    def __init__(self, params: BirchParams):
        self.model = Birch(
            branching_factor=params.branching_factor,
            n_clusters=params.n_clusters,
            threshold=params.threshold,
        )

    def fit(self, log_features: pd.DataFrame):
        """
        Train BIRCH model
        :param log_features: pd.DataFrame: log features for training.
        :return:
        """
        log_features = np.ascontiguousarray(log_features)
        self.model.partial_fit(log_features)

    def predict(self, log_features: pd.DataFrame) -> pd.Series:
        """
        Inference using trained BIRCH model
        :param log_features: log features for inference
        :return: pd.Series: series of log cluster labels
        """
        log_features_carray = np.ascontiguousarray(log_features)

        res = self.model.predict(log_features_carray)
        return pd.Series(res, index=log_features.index)
