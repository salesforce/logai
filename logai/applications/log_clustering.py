#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd

from logai.algorithms.clustering_algo.dbscan import DBSCAN
from logai.algorithms.clustering_algo.kmeans import KMeans
from logai.analysis.clustering import Clustering
from logai.applications.application_interfaces import WorkFlowConfig
from logai.dataloader.data_loader import FileDataLoader
from logai.dataloader.openset_data_loader import OpenSetDataLoader
from logai.information_extraction.categorical_encoder import CategoricalEncoder
from logai.information_extraction.feature_extractor import FeatureExtractor
from logai.information_extraction.log_parser import LogParser
from logai.information_extraction.log_vectorizer import LogVectorizer
from logai.preprocess.preprocessor import Preprocessor
from logai.utils import constants
from logai.utils.functions import pad


class LogClustering:
    """
    Clustering Application class defines log clustering workflow.
    It includes which algorithm to use.
    """

    def __init__(self, config):
        self.config = config
        self._loglines = pd.DataFrame()
        self._timestamps = pd.DataFrame()
        self._attributes = pd.DataFrame()
        self._feature_df = pd.DataFrame()
        self._clusters = pd.DataFrame()
        self.MAX_LEN = 100
        pass

    @property
    def clusters(self):
        return self._clusters

    @property
    def attributes(self):
        return self._attributes

    @property
    def timestamps(self):
        return self._timestamps

    @property
    def event_index(self):
        return self._index

    @property
    def loglines(self):
        return self._loglines

    @property
    def logline_with_clusters(self):
        event_cluster_df = pd.concat(
            (self.clusters, self.loglines, self.attributes, self.timestamps), axis=1
        )
        return event_cluster_df

    def execute(self):

        # Load data
        logrecord = self._load_data()

        # Preprocessor cleans the loglines
        logline = logrecord.body[constants.LOGLINE_NAME]

        self._loglines = logline
        self._timestamps = logrecord.timestamp

        preprocessor = Preprocessor(self.config.preprocessor_config)
        preprocessed_loglines, _ = preprocessor.clean_log(logline)

        # Parsing
        parser = LogParser(self.config.log_parser_config)
        parsed_results = parser.parse(preprocessed_loglines.dropna())

        parsed_loglines = parsed_results[constants.PARSED_LOGLINE_NAME]

        # Vectorization
        vectorizor = LogVectorizer(self.config.log_vectorizer_config)
        vectorizor.fit(parsed_loglines)

        # Log vector is a pandas.Series
        log_vectors = vectorizor.transform(parsed_loglines)

        # Categorical Encoding
        encoder = CategoricalEncoder(self.config.categorical_encoder_config)

        self._attributes = logrecord.attributes.astype(str)

        attributes = encoder.fit_transform(logrecord.attributes)
        attributes.columns = logrecord.attributes.columns

        padded_log_vectors = log_vectors.apply(pad, max_len=self.MAX_LEN)

        feature_for_clustering = pd.DataFrame(
            [vec for vec in padded_log_vectors], index=padded_log_vectors.index
        )

        feature_for_clustering = feature_for_clustering.join(attributes)

        log_clustering = Clustering(self.config.clustering_config)

        log_clustering.fit(feature_for_clustering)

        self._clusters = (
            log_clustering.predict(feature_for_clustering)
            .astype(str)
            .rename("cluster_id")
        )
        self._index = self._clusters.index

        return

    def _load_data(self):
        if self.config.open_set_data_loader_config is not None:
            dataloader = OpenSetDataLoader(self.config.open_set_data_loader_config)
            logrecord = dataloader.load_data()
        elif self.config.data_loader_config is not None:
            dataloader = FileDataLoader(self.config.data_loader_config)
            logrecord = dataloader.load_data()
        else:
            raise ValueError(
                "data_loader_config or open_set_data_loader_config is needed to load data."
            )
        return logrecord
