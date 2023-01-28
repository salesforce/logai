#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd
from attr import dataclass
from sklearn.cluster import KMeans

from logai.algorithms.algo_interfaces import ClusteringAlgo
from logai.config_interfaces import Config
from logai.algorithms.factory import factory


@dataclass
class KMeansParams(Config):
    n_clusters: int = 8
    init: str = "k-means++"
    n_init: int = 10
    max_iter: int = 300
    tol: float = 1e-4
    verbose: int = 0
    random_state: int = None
    copy_x: bool = True
    algorithm: str = "auto"


@factory.register("clustering", "kmeans", KMeansParams)
class KMeansAlgo(ClusteringAlgo):
    """
    Implement K-means for log clustering
    """

    def __init__(self, params: KMeansParams):
        self.model = KMeans(
            n_clusters=params.n_clusters,
            init=params.init,
            n_init=params.n_init,
            max_iter=params.max_iter,
            tol=params.tol,
            verbose=params.verbose,
            random_state=params.random_state,
            copy_x=params.copy_x,
            algorithm=params.algorithm,
        )

    def fit(self, log_features: pd.DataFrame):
        """
        Fit K-means model.
        :param log_features: log features for training
        :return:
        """
        self.model.fit(log_features)

    def predict(self, log_features: pd.DataFrame) -> pd.Series:
        """
        Inference using trained K-means model.
        :param log_features: log features for inference.
        :return: pd.Series: series of cluster labels.
        """
        res = self.model.predict(log_features)
        return pd.Series(res, index=log_features.index)
