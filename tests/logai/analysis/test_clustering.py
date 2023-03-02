#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd

from logai.algorithms.clustering_algo.birch import BirchAlgo
from logai.algorithms.clustering_algo.dbscan import DbScanAlgo
from logai.algorithms.clustering_algo.kmeans import KMeansAlgo
from logai.analysis.clustering import ClusteringConfig, Clustering

from tests.logai.test_utils.fixtures import log_features


class TestClustering:
    def setup(self):
        self.supported_algos = ['kmeans', 'dbscan', 'birch']

    def test_creating_models_default_params_kmeans(self):
        config = ClusteringConfig(algo_name='kmeans')
        clustering = Clustering(config)
        assert isinstance(clustering, Clustering), 'Clustering analyzer creation failed'
        assert isinstance(clustering.model, KMeansAlgo), 'Model instance does not match definition'

    def test_creating_models_default_params_dbscan(self):
        config = ClusteringConfig(algo_name='dbscan')
        clustering = Clustering(config)
        assert isinstance(clustering, Clustering), 'Clustering analyzer creation failed'
        assert isinstance(clustering.model, DbScanAlgo), 'Model instance does not match definition'

    def test_creating_models_default_params_birch(self):
        config = ClusteringConfig(algo_name='birch')
        clustering = Clustering(config)
        assert isinstance(clustering, Clustering), 'Clustering analyzer creation failed'
        assert isinstance(clustering.model, BirchAlgo), 'Model instance does not match definition'

    def test_fit_predict_default(self, log_features):
        for algo in self.supported_algos:
            config = ClusteringConfig(algo_name=algo)
            clustering = Clustering(config)
            clustering.fit(log_features)
            labels = clustering.predict(log_features)

            assert isinstance(labels, pd.Series), 'labels are not pd.Series'
