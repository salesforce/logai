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
    """Parameters of the KMeans Clustering algorithm. For more details on the parameters see 
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html.

    :param n_clusters: The number of clusters to form as well as the number of centroids to generate.
    :param init: Method for initialization, i.e., ``{‘k-means++’, ‘random’}``.
    :param n_init: Number of times the k-means algorithm is run with different centroid seeds.
    :param max_iter: Maximum number of iterations of the k-means algorithm for a single run.
    :param tol: Relative tolerance with regards to Frobenius norm of the difference in the cluster
        centers of two consecutive iterations to declare convergence.
    :param verbose: Verbosity mode.
    :param random_state: Determines random number generation for centroid initialization.
    :param copy_x: If copy_x is True (default), then the original data is not modified.
        If False, the original data is modified, and put back before the function returns.
    :param algorithm: K-means algorithm to use, i.e., ``{“lloyd”, “elkan”, “auto”, “full”}``.
    """
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
    K-means algorithm for log clustering. This is a wrapper class for K-Means clustering method from
    scikit-learn library https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html.
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
        Fits a K-means model.

        :param log_features: The log features for training
        """
        self.model.fit(log_features)

    def predict(self, log_features: pd.DataFrame) -> pd.Series:
        """
        Predicts using trained K-means model.

        :param log_features: The log features for inference.
        :return: A pandas series of cluster labels.
        """
        res = self.model.predict(log_features)
        return pd.Series(res, index=log_features.index)
