#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from attr import dataclass

from logai.analysis.anomaly_detector import AnomalyDetectionConfig
from logai.analysis.nn_anomaly_detector import NNAnomalyDetectionConfig
from logai.analysis.clustering import ClusteringConfig
from logai.config_interfaces import Config
from logai.dataloader.data_loader import DataLoaderConfig
from logai.dataloader.openset_data_loader import OpenSetDataLoaderConfig
from logai.information_extraction.categorical_encoder import CategoricalEncoderConfig
from logai.information_extraction.feature_extractor import FeatureExtractorConfig
from logai.information_extraction.log_parser import LogParserConfig
from logai.information_extraction.log_vectorizer import VectorizerConfig
from logai.preprocess.partitioner import PartitionerConfig
from logai.preprocess.openset_partitioner import OpenSetPartitionerConfig
from logai.preprocess.preprocessor import PreprocessorConfig


@dataclass
class WorkFlowConfig(Config):
    data_loader_config: object = None
    open_set_data_loader_config: object = None
    preprocessor_config: object = None

    log_parser_config: object = None
    log_vectorizer_config: object = None
    partitioner_config: object = None
    open_set_partitioner_config: object = None
    categorical_encoder_config: object = None
    feature_extractor_config: object = None
    anomaly_detection_config: object = None
    nn_anomaly_detection_config: object = None 
    clustering_config: object = None
    workflow_config: object = None

    def from_dict(self, config_dict):
        if not config_dict:
            return
        for field in self.__dict__:
            if field in config_dict:
                setattr(self, field, config_dict[field])

        if self.data_loader_config:
            self.data_loader_config = \
                DataLoaderConfig.from_dict(self.data_loader_config)

        if self.open_set_data_loader_config:
            self.open_set_data_loader_config = \
                OpenSetDataLoaderConfig.from_dict(self.open_set_data_loader_config)

        if self.preprocessor_config:
            self.preprocessor_config = \
                PreprocessorConfig.from_dict(self.preprocessor_config)

        if self.partitioner_config:
            self.partitioner_config = \
                PartitionerConfig.from_dict(self.partitioner_config)

        if self.open_set_partitioner_config:
            self.open_set_partitioner_config = \
                OpenSetPartitionerConfig.from_dict(self.open_set_partitioner_config)

        if self.log_parser_config:
            self.log_parser_config = \
                LogParserConfig.from_dict(self.log_parser_config)

        if self.log_vectorizer_config:
            self.log_vectorizer_config = \
                VectorizerConfig.from_dict(self.log_vectorizer_config)

        if self.feature_extractor_config:
            self.feature_extractor_config = \
                FeatureExtractorConfig.from_dict(self.feature_extractor_config)

        if self.categorical_encoder_config:
            self.categorical_encoder_config = \
                CategoricalEncoderConfig.from_dict(self.categorical_encoder_config)

        if self.anomaly_detection_config:
            self.anomaly_detection_config = \
                AnomalyDetectionConfig.from_dict(self.anomaly_detection_config)

        if self.nn_anomaly_detection_config:
            self.nn_anomaly_detection_config = \
                NNAnomalyDetectionConfig.from_dict(self.nn_anomaly_detection_config)

        if self.clustering_config:
            self.clustering_config = \
                ClusteringConfig.from_dict(self.clustering_config)
