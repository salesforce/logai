#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from attr import dataclass
from datasets import Dataset as HFDataset

from logai.algorithms.vectorization_algo.forecast_nn import ForecastNNVectorizedDataset
from logai.config_interfaces import Config
from logai.algorithms.factory import factory


@dataclass
class NNAnomalyDetectionConfig(Config):
    algo_name: str = "logbert"
    algo_params: object = None
    custom_params: object = None

    def from_dict(self, config_dict):
        super().from_dict(config_dict)
        self.algo_params = factory.get_config(
            "detection", self.algo_name.lower(), self.algo_params)


class NNAnomalyDetector:
    def __init__(self, config: NNAnomalyDetectionConfig):
        """
        Init the anomaly detector with proper configuration. If no config provided, use default
        :param config: AnomalyDetectionConfig
        """
        self.anomaly_detector = factory.get_algorithm(
            "detection", config.algo_name.lower(), config)

    def fit(
            self,
            train_data: ForecastNNVectorizedDataset or HFDataset,
            dev_data: ForecastNNVectorizedDataset or HFDataset
    ):
        """
        Fit model
        :param train_data:
        :return:
        """
        return self.anomaly_detector.fit(train_data, dev_data)

    def predict(self, test_data: ForecastNNVectorizedDataset or HFDataset):
        """
        Predict for input
        :param test_data:
        :return:
        """
        return self.anomaly_detector.predict(test_data)
