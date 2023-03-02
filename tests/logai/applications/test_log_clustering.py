#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import json
import os

from logai.algorithms.parsing_algo.drain import DrainParams
from logai.analysis.clustering import ClusteringConfig
from logai.applications.application_interfaces import WorkFlowConfig
from logai.applications.log_clustering import LogClustering
from logai.dataloader.data_loader import DataLoaderConfig
from logai.information_extraction.categorical_encoder import CategoricalEncoderConfig
from logai.information_extraction.log_parser import LogParserConfig
from logai.information_extraction.log_vectorizer import VectorizerConfig
from logai.preprocess.preprocessor import PreprocessorConfig

TEST_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_data/HealthApp_2000.log')
TEST_HDFS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_data/HDFS/HDFS_5000.log')
TEST_BGL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_data/BGL_5000.log')


class TestLogClustering:
    def setup(self):
        self.config = WorkFlowConfig(
            data_loader_config=DataLoaderConfig(
                filepath=TEST_LOG_PATH,
                log_type='csv',
                dimensions=dict({
                    "timestamp": ["Timestamps"],
                    "attributes": ["Action", "ID"],
                    "body": ["Details"]
                }),
                reader_args={
                    "header": None,
                    "sep": "|",
                    "on_bad_lines": "skip",
                    "names": ["Timestamps", "Action", "ID", "Details"],
                },
                infer_datetime=True,
                datetime_format='%Y%m%d-%H:%M:%S:%f',
            ),
            preprocessor_config=PreprocessorConfig(
                custom_delimiters_regex=None
            ),
            log_parser_config=LogParserConfig(
                parsing_algorithm='drain',
                parsing_algo_params=DrainParams(
                    sim_th=0.4
                )
            ),
            log_vectorizer_config=VectorizerConfig(
                algo_name="word2vec",
            ),
            categorical_encoder_config=CategoricalEncoderConfig(
                name="label_encoder",
            ),
            clustering_config=ClusteringConfig(
                algo_name='kmeans',
            ),
        )

    def test_execute(self):
        app = LogClustering(self.config)
        app.execute()
        print(app.clusters)

    def test_logline_with_clusters(self):
        app = LogClustering(self.config)
        app.execute()
        res = app.logline_with_clusters.dropna()
        print(res.shape[0])

    def test_hdfs_data(self):
        json_config = """{
  "open_set_data_loader_config": {
    "dataset_name": "HDFS"
  },
  "preprocessor_config": {
      "custom_delimiters_regex":[]
  },
  "log_parser_config": {
    "parsing_algorithm": "drain",
    "parsing_algo_params": {
      "sim_th": 0.5,
      "depth": 5
    }
  },
  "log_vectorizer_config": {
      "algo_name": "word2vec"
  },
  "categorical_encoder_config": {
      "name": "label_encoder"
  },
      "anomaly_detection_config": {
      "algo_name": "one_class_svm"
  },
  "clustering_config": {
      "algo_name": "kmeans",
      "algo_params": {
          "n_clusters": 7
      }
  }
}
"""
        config = json.loads(json_config)
        print(config)
        workflow_config = WorkFlowConfig.from_dict(config)
        print(workflow_config)
        workflow_config.open_set_data_loader_config.filepath = TEST_HDFS_PATH
        app = LogClustering(workflow_config)
        app.execute()

    def test_bgl_data(self):
        json_config = """{
  "open_set_data_loader_config": {
    "dataset_name": "BGL"
  },
  "preprocessor_config": {
      "custom_delimiters_regex":[]
  },
  "log_parser_config": {
    "parsing_algorithm": "drain",
    "parsing_algo_params": {
      "sim_th": 0.5,
      "depth": 5
    }
  },
  "log_vectorizer_config": {
      "algo_name": "tfidf"
  },
  "categorical_encoder_config": {
      "name": "label_encoder"
  },
      "anomaly_detection_config": {
      "algo_name": "one_class_svm"
  },
  "clustering_config": {
      "algo_name": "kmeans",
      "algo_params": {
          "n_clusters": 7
      }
  }
}
"""
        config = json.loads(json_config)
        print(config)
        workflow_config = WorkFlowConfig.from_dict(config)
        print(workflow_config)
        workflow_config.open_set_data_loader_config.filepath = TEST_BGL_PATH
        app = LogClustering(workflow_config)
        app.execute()
