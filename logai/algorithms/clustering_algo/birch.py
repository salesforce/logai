#
# Copyright (c) 2023 Salesforce.com, inc.
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
    """Parameters for Birch Clustering Algo. For more details on the parameters, see 
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html.

    :param branching_factor: Maximum number of CF subclusters in each node.
    :param n_clusters: Number of clusters after the final clustering step, which treats the subclusters
        from the leaves as new samples.
    :param threshold: The radius of the subcluster obtained by merging a new sample and the closest
        subcluster should be lesser than the threshold.
    """
    branching_factor: int = 50
    n_clusters: int = None
    threshold: float = 1.5


@factory.register("clustering", "birch", BirchParams)
class BirchAlgo(ClusteringAlgo):
    """
    BIRCH algorithm for log clustering. This is a wrapper class for the Birch Clustering algorithm in scikit-learn 
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html.
    """

    def __init__(self, params: BirchParams):
        self.model = Birch(
            branching_factor=params.branching_factor,
            n_clusters=params.n_clusters,
            threshold=params.threshold,
        )

    def fit(self, log_features: pd.DataFrame):
        """
        Trains a BIRCH model.

        :param log_features: The log features for training.
        """
        log_features = np.ascontiguousarray(log_features)
        self.model.partial_fit(log_features)

    def predict(self, log_features: pd.DataFrame) -> pd.Series:
        """
        Predicts using trained BIRCH model.

        :param log_features: The log features for inference.
        :return: A pandas series of log cluster labels.
        """
        log_features_carray = np.ascontiguousarray(log_features)

        res = self.model.predict(log_features_carray)
        return pd.Series(res, index=log_features.index)
