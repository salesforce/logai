#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import json
import yaml

from logai.applications.application_interfaces import WorkFlowConfig
from logai.applications.log_clustering import LogClustering as LogClusteringApp
from .utils import ParamInfoMixin


class Clustering(ParamInfoMixin):
    algorithms = {
        "birch": ("logai.algorithms.clustering_algo.birch", "BirchAlgo", "BirchParams"),
        "dbscan": (
            "logai.algorithms.clustering_algo.dbscan",
            "DbScanAlgo",
            "DbScanParams",
        ),
        "kmeans": (
            "logai.algorithms.clustering_algo.kmeans",
            "KMeansAlgo",
            "KMeansParams",
        ),
    }

    def __init__(self):
        self.app = None
        self.attributes = None
        pass

    # TODO: merge with PD and move to utils
    def json_to_config(self, json_config):
        config = json.loads(json_config)
        workflow_config = WorkFlowConfig.from_dict(config)
        return workflow_config

    # TODO: merge with PD and move to utils
    def yaml_to_config(self, yaml_config):
        config = yaml.safe_load(yaml_config)
        workflow_config = WorkFlowConfig.from_dict(config)
        return workflow_config

    def execute_clustering(self, config: WorkFlowConfig):
        """
        This function executes the auto log parsing application.
        :param config: WorkFlowConfig
            Sample config:
            -------------------
            config = WorkFlowConfig(
                data_loader_config=DataLoaderConfig(
                    filepath=LOG_PATH,
                    log_type='csv',
                    dimensions=dict({
                        "attributes": ["Action", "ID"],
                        "body": ["Details"]
                    }),
                    reader_arg={
                        "header": None,
                        "sep": "|",
                        "on_bad_lines": "skip",
                        "names": ["Timestamps", "Action", "ID", "Details"]
                    }
                ),
                feature_extractor_config=FeatureExtractorConfig(),
                preprocessor_config=PreprocessorConfig(
                    custom_delimiters_regex=None
                ),
                log_parser_config=LogParserConfig(
                    parsing_algorithm='drain',
                    parsing_algo_params=DrainParams(
                        sim_th=0.4
                    )
                ),
            )
            -----------------------
        :return:
        """

        self.app = LogClusteringApp(config)
        self.app.execute()

        return

    def get_attributes(self):
        return self.app.attributes

    def get_unique_clusters(self):
        count_table = self.app.clusters.value_counts()

        return count_table.to_dict()

    def get_loglines(self, cluster_id):
        df = self.app.logline_with_clusters
        loglines = df.loc[df["cluster_id"].astype(str) == cluster_id]
        return loglines

    @property
    def result_table(self):
        return self.app.logline_with_clusters
