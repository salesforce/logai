#
# Copyright (c) 2022 Salesforce.com, inc.
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
    algo_name: str = "one_class_svm"
    algo_params: object = None
    custom_params: object = None

    def from_dict(self, config_dict):
        super().from_dict(config_dict)
        self.algo_params = factory.get_config(
            "detection", self.algo_name.lower(), self.algo_params)


class AnomalyDetector:
    def __init__(self, config: AnomalyDetectionConfig):
        """
        Init the anomaly detector with proper configuration. If no config provided, use default
        :param config: AnomalyDetectionConfig
        """
        self.anomaly_detector = factory.get_algorithm(
            "detection", config.algo_name.lower(), config)

    def fit(self, log_features: pd.DataFrame):
        """
        Fit model
        :param log_features:
        :return:
        """
        return self.anomaly_detector.fit(log_features)

    def predict(self, log_features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict for input
        :param log_features:
        :return:
        """
        return self.anomaly_detector.predict(log_features)
