#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import os

import pandas as pd

from gui.demo.log_clustering import Clustering
from logai.algorithms.parsing_algo.drain import DrainParams
from logai.analysis.clustering import ClusteringConfig
from logai.applications.application_interfaces import WorkFlowConfig
from logai.dataloader.data_loader import DataLoaderConfig
from logai.dataloader.openset_data_loader import OpenSetDataLoaderConfig
from logai.information_extraction.categorical_encoder import CategoricalEncoderConfig
from logai.information_extraction.feature_extractor import FeatureExtractorConfig
from logai.information_extraction.log_parser import LogParserConfig
from logai.information_extraction.log_vectorizer import VectorizerConfig
from logai.preprocess.preprocessor import PreprocessorConfig

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class TestClustering:
    def setup(self):
        self.config = WorkFlowConfig(
            open_set_data_loader_config=OpenSetDataLoaderConfig(
                dataset_name="HealthApp",
                filepath=os.path.join(DIR, "HealthApp_2000.log")
            ),
            feature_extractor_config=FeatureExtractorConfig(
                group_by_time="1min",
                group_by_category=["Action", "ID"]
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
                algo_name="kmeans",
            )
        )

        return

    def test_execute_app(self):
        processor = Clustering()
        processor.execute_clustering(self.config)

        df = pd.DataFrame.from_dict(processor.get_unique_clusters(), orient="index")
        df.index.name = "Cluster"
        df.columns = ['Size']

        print(processor.get_loglines("0"))
        return
