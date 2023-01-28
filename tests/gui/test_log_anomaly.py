#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import os

import pandas as pd

from gui.demo.log_anomaly import LogAnomaly
from logai.algorithms.parsing_algo.drain import DrainParams
from logai.analysis.anomaly_detector import AnomalyDetectionConfig
from logai.applications.application_interfaces import WorkFlowConfig
from logai.dataloader.openset_data_loader import OpenSetDataLoaderConfig
from logai.information_extraction.categorical_encoder import CategoricalEncoderConfig
from logai.information_extraction.feature_extractor import FeatureExtractorConfig
from logai.information_extraction.log_parser import LogParserConfig
from logai.information_extraction.log_vectorizer import VectorizerConfig
from logai.preprocess.preprocessor import PreprocessorConfig

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class TestLogAnomaly:
    def setup(self):
        self.config = WorkFlowConfig(
            open_set_data_loader_config=OpenSetDataLoaderConfig(
                dataset_name="HealthApp",
                filepath=os.path.join(DIR, "HealthApp_2000.log")
            ),
            feature_extractor_config=FeatureExtractorConfig(
                group_by_time="1s",
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
            anomaly_detection_config=AnomalyDetectionConfig(
                algo_name="dbl",
            )
        )

        return

    def test_execute_app(self):
        processor = LogAnomaly()
        processor.execute_anomaly_detection(self.config)

        attrs = processor.get_attributes()

        pass

    def test_anomalies(self):
        processor = LogAnomaly()
        processor.execute_anomaly_detection(self.config)

        attributes = processor.get_attributes()

        attr = {
                "Action": attributes['Action'][0],
                "ID": attributes['ID'][0]
        }

        df = processor.get_results(attr)

        print(df.head(5))
        return



