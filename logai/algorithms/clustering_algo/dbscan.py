#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd
from attr import dataclass
from sklearn.cluster import DBSCAN

from logai.algorithms.algo_interfaces import ClusteringAlgo
from logai.config_interfaces import Config
from logai.algorithms.factory import factory


@dataclass
class DbScanParams(Config):
    """Parameters for DBScan based clustering algorithm. For more details on parameters see 
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html.

    :param eps: The maximum distance between two samples for one to be considered as in the
        neighborhood of the other.
    :param min_samples: The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point.
    :param metric: The metric to use when calculating distance between instances in a feature array.
    :param metric_params: Additional keyword arguments for the metric function.
    :param algorithm: The algorithm to be used by the NearestNeighbors module to compute pointwise
        distances and find nearest neighbors, i.e., ``{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}``.
    :param leaf_size: Leaf size passed to BallTree or cKDTree.
    :param p: The power of the Minkowski metric to be used to calculate distance between points.
    :param n_jobs: The number of parallel jobs to run.
    """
    eps: float = 0.3
    min_samples: int = 10
    metric: str = "euclidean"
    metric_params: object = None
    algorithm: str = "auto"
    leaf_size: int = 30
    p: float = None
    n_jobs: int = None


@factory.register("clustering", "dbscan", DbScanParams)
class DbScanAlgo(ClusteringAlgo):
    """
    DBSCAN algorithm for log clustering. This is a wrapper class for DBScan based from scikit-learn library 
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    """

    def __init__(self, params: DbScanParams):
        self.model = DBSCAN(
            eps=params.eps,
            min_samples=params.min_samples,
            metric=params.metric,
            metric_params=params.metric_params,
            algorithm=params.algorithm,
            leaf_size=params.leaf_size,
            p=params.p,
            n_jobs=params.n_jobs,
        )

    def fit(self, log_features: pd.DataFrame):
        """
        Trains a DBSCAN model.

        :param log_features: The log features as training data.
        """
        self.model.fit(log_features)

    def predict(self, log_features: pd.DataFrame) -> pd.Series:
        """
        Predicts using the trained DBSCAN model.

        :param log_features: The log features for inference.
        :return: A pandas series of cluster labels.
        """
        res = self.model.fit_predict(log_features)
        return pd.Series(res, index=log_features.index)
