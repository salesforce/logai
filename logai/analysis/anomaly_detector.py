#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd
import logai.algorithms.anomaly_detection_algo

from attr import dataclass
from logai.config_interfaces import Config
from logai.algorithms.factory import factory


@dataclass
class AnomalyDetectionConfig(Config):
    """Config class for AnomalyDetector.

    :param algo_name: The algorithm name.
    :param algo_params: The algorithm parameters.
    :param custom_params: Additional customized parameters.
    """
    algo_name: str = "one_class_svm"
    algo_params: object = None
    custom_params: object = None

    @classmethod
    def from_dict(cls, config_dict):
        config = super(AnomalyDetectionConfig, cls).from_dict(config_dict)
        config.algo_params = factory.get_config(
            "detection", config.algo_name.lower(), config.algo_params
        )
        return config


class AnomalyDetector:
    def __init__(self, config: AnomalyDetectionConfig):
        """
        Initializes the anomaly detector with proper configuration. If no config provided, use default.

        :param config: A config object for anomaly detection.
        """
        self.anomaly_detector = factory.get_algorithm(
            "detection", config.algo_name.lower(), config
        )

    def fit(self, log_features: pd.DataFrame):
        """
        Trains an anomaly detection given the training dataset.

        :param log_features: The training dataset.
        """
        return self.anomaly_detector.fit(log_features)

    def predict(self, log_features: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts anomalies given the test dataset.

        :param log_features: The test dataset.
        :return: A pandas dataframe containing the prediction results.
        """
        return self.anomaly_detector.predict(log_features)
