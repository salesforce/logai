#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd
from attr import dataclass

from logai.algorithms.clustering_algo.birch import BirchParams, BirchAlgo
from logai.algorithms.clustering_algo.dbscan import DbScanAlgo, DbScanParams
from logai.algorithms.clustering_algo.kmeans import KMeansAlgo, KMeansParams
from logai.config_interfaces import Config


@dataclass
class ClusteringConfig(Config):
    algo_name: str = "dbscan"
    algo_params: object = None
    custom_params: object = None

    def from_dict(self, config_dict):
        super().from_dict(config_dict)

        if self.algo_name.lower() == "dbscan":
            params = DbScanParams()
            params.from_dict(self.algo_params)
            self.algo_params = params

        elif self.algo_name.lower() == "kmeans":
            params = KMeansParams()
            params.from_dict(self.algo_params)
            self.algo_params = params

        elif self.algo_name.lower() == "birch":
            params = BirchParams()
            params.from_dict(self.algo_params)
            self.algo_params = params

        else:
            raise RuntimeError()

        return


class Clustering:
    """
    Clustering Application class defines log clustering workflow.
    It includes which algorithm to use.
    """

    def __init__(self, config: ClusteringConfig):
        if config.algo_name.lower() == "dbscan":
            self.model = DbScanAlgo(
                config.algo_params if config.algo_params else DbScanParams()
            )
        elif config.algo_name.lower() == "kmeans":
            self.model = KMeansAlgo(
                config.algo_params if config.algo_params else KMeansParams()
            )
        elif config.algo_name.lower() == "birch":
            self.model = BirchAlgo(
                config.algo_params if config.algo_params else BirchParams()
            )
        else:
            raise RuntimeError(
                "Clustering Algorithm {} is not defined".format(config.algo_name)
            )

    def fit(self, log_features: pd.DataFrame):
        self.model.fit(log_features)
        return

    def predict(self, log_features: pd.DataFrame) -> pd.Series:
        return self.model.predict(log_features)
