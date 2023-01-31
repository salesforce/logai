#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from datasets import Dataset as HFDataset
from logai.algorithms.vectorization_algo.forecast_nn import ForecastNNVectorizedDataset
from logai.analysis.anomaly_detector import AnomalyDetectionConfig
from logai.algorithms.factory import factory

NNAnomalyDetectionConfig = AnomalyDetectionConfig


class NNAnomalyDetector:
    def __init__(self, config: NNAnomalyDetectionConfig):
        """
        Initializes the anomaly detector with proper configuration. If no config provided, use default.

        :param config: A config object for anomaly detection.
        """
        self.anomaly_detector = factory.get_algorithm(
            "detection", config.algo_name.lower(), config
        )

    def fit(
        self,
        train_data: ForecastNNVectorizedDataset or HFDataset,
        dev_data: ForecastNNVectorizedDataset or HFDataset,
    ):
        """
        Trains an anomaly detection given the training and validation datasets.

        :param train_data: The training dataset.
        :param dev_data: The validation dataset
        """
        return self.anomaly_detector.fit(train_data, dev_data)

    def predict(self, test_data: ForecastNNVectorizedDataset or HFDataset):
        """
        Predicts anomalies given the test dataset.

        :param test_data: The test dataset.
        :return: A pandas dataframe containing the prediction results.
        """
        return self.anomaly_detector.predict(test_data)
