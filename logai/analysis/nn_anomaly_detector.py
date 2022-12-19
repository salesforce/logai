#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd
from attr import dataclass


from logai.algorithms.vectorization_algo.forecast_nn import ForecastNNVectorizedDataset

from datasets import Dataset as HFDataset

from logai.algorithms.anomaly_detection_algo.logbert import LogBERT, LogBERTConfig
from logai.algorithms.anomaly_detection_algo.forecast_nn import (
    ForecastBasedLSTM,
    LSTMParams,
    ForecastBasedCNN,
    CNNParams,
    ForecastBasedTransformer,
    TransformerParams,
)

from logai.config_interfaces import Config


@dataclass
class NNAnomalyDetectionConfig(Config):
    algo_name: str = "one_class_svm"
    algo_params: object = None
    custom_params: object = None

    def from_dict(self, config_dict):
        super().from_dict(config_dict)

        if self.algo_name.lower() == "logbert":
            params = LogBERTConfig()
            params.from_dict(self.algo_params)
            self.algo_params = params

        elif self.algo_name.lower() == "lstm":
            params = LSTMParams()
            params.from_dict(self.algo_params)
            self.algo_params = params

        elif self.algo_name.lower() == "cnn":
            params = CNNParams()
            params.from_dict(self.algo_params)
            self.algo_params = params

        elif self.algo_name.lower() == "transformer":
            params = TransformerParams()
            params.from_dict(self.algo_params)
            self.algo_params = params

        else:
            raise RuntimeError()

        return


class NNAnomalyDetector:
    def __init__(self, config: NNAnomalyDetectionConfig):
        """
        Init the anomaly detector with proper configuration. If no config provided, use default
        :param config: AnomalyDetectionConfig
        """
        if config.algo_name.lower() == "logbert":
            self.anomaly_detector = LogBERT(
                config.algo_params if config.algo_params else LogBERTConfig()
            )

        elif config.algo_name.lower() == "lstm":
            self.anomaly_detector = ForecastBasedLSTM(
                config.algo_params if config.algo_params else LSTMParams()
            )

        elif config.algo_name.lower() == "cnn":
            self.anomaly_detector = ForecastBasedCNN(
                config.algo_params if config.algo_params else CNNParams()
            )

        elif config.algo_name.lower() == "transformer":
            self.anomaly_detector = ForecastBasedTransformer(
                config.algo_params if config.algo_params else TransformerParams()
            )

        else:
            raise RuntimeError(
                "Anomaly detection algorithm {} is not defined".format(config.algo_name)
            )

        return

    def fit(self, train_data: ForecastNNVectorizedDataset or HFDataset, dev_data: ForecastNNVectorizedDataset or HFDataset):
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
