#
# Copyright (c) 2023 Salesforce.com, inc.
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
    """config class for end to end workflow.
    
    :param data_loader_config: A config object for data loader.
    :param open_set_data_loader_config: A config object for data loader for opensource public log datasets.
    :param preprocessor_config: A config object for log preprocessor.
    :param log_parser_config: A config object for log parser.
    :param log_vectorizer_config: A config object for log vectorizer.
    :param partitioner_config: A config object for log partitioner.
    :param open_set_partitioner_config: A config object for log partitioner for opensource public log datasets.
    :param categorical_encoder_config: A config object for categorical encoder of log data.
    :param feature_extractor_config: A config object for log feature extractor.
    :param anomaly_detection_config: A config object for log anomaly detector.
    :param nn_anomaly_detection_config: A config object for neural anomaly detector.
    :param clustering_config: A config object for log clustering algorithm.
    """
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

    @classmethod
    def from_dict(cls, config_dict):
        config = super(WorkFlowConfig, cls).from_dict(config_dict)

        if config.data_loader_config:
            config.data_loader_config = DataLoaderConfig.from_dict(
                config.data_loader_config
            )

        if config.open_set_data_loader_config:
            config.open_set_data_loader_config = OpenSetDataLoaderConfig.from_dict(
                config.open_set_data_loader_config
            )

        if config.preprocessor_config:
            config.preprocessor_config = PreprocessorConfig.from_dict(
                config.preprocessor_config
            )

        if config.partitioner_config:
            config.partitioner_config = PartitionerConfig.from_dict(
                config.partitioner_config
            )

        if config.open_set_partitioner_config:
            config.open_set_partitioner_config = OpenSetPartitionerConfig.from_dict(
                config.open_set_partitioner_config
            )

        if config.log_parser_config:
            config.log_parser_config = LogParserConfig.from_dict(
                config.log_parser_config
            )

        if config.log_vectorizer_config:
            config.log_vectorizer_config = VectorizerConfig.from_dict(
                config.log_vectorizer_config
            )

        if config.feature_extractor_config:
            config.feature_extractor_config = FeatureExtractorConfig.from_dict(
                config.feature_extractor_config
            )

        if config.categorical_encoder_config:
            config.categorical_encoder_config = CategoricalEncoderConfig.from_dict(
                config.categorical_encoder_config
            )

        if config.anomaly_detection_config:
            config.anomaly_detection_config = AnomalyDetectionConfig.from_dict(
                config.anomaly_detection_config
            )

        if config.nn_anomaly_detection_config:
            config.nn_anomaly_detection_config = NNAnomalyDetectionConfig.from_dict(
                config.nn_anomaly_detection_config
            )

        if config.clustering_config:
            config.clustering_config = ClusteringConfig.from_dict(
                config.clustering_config
            )

        return config
