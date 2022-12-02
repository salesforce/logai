#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd
from attr import dataclass

from logai.algorithms.anomaly_detection_algo.dbl import DBLDetectorParams, DBLDetector
from logai.algorithms.anomaly_detection_algo.ets import ETSDetectorParams, ETSDetector
from logai.algorithms.anomaly_detection_algo.isolation_forest import (
    IsolationForestParams,
    IsolationForestDetector,
)
from logai.algorithms.anomaly_detection_algo.local_outlier_factor import (
    LOFDetector,
    LOFParams,
)
from logai.algorithms.anomaly_detection_algo.one_class_svm import (
    OneClassSVMDetector,
    OneClassSVMParams,
)
from logai.algorithms.anomaly_detection_algo.distribution_divergence import (
    DistributionDivergence,
    DistributionDivergenceParams,
)
from logai.config_interfaces import Config


@dataclass
class AnomalyDetectionConfig(Config):
    algo_name: str = "one_class_svm"
    algo_params: object = None
    custom_params: object = None

    def from_dict(self, config_dict):
        super().from_dict(config_dict)

        if self.algo_name.lower() == "one_class_svm":
            params = OneClassSVMParams()
            params.from_dict(self.algo_params)
            self.algo_params = params

        elif self.algo_name.lower() == "isolation_forest":
            params = IsolationForestParams()
            params.from_dict(self.algo_params)
            self.algo_params = params

        elif self.algo_name.lower() == "lof":
            params = LOFParams()
            params.from_dict(self.algo_params)
            self.algo_params = params

        elif self.algo_name.lower() == "distribution_divergence":
            params = DistributionDivergenceParams()
            params.from_dict(self.algo_params)
            self.algo_params = params

        elif self.algo_name.lower() == "dbl":
            params = DBLDetectorParams()
            params.from_dict(self.algo_params)
            self.algo_params = params

        elif self.algo_name.lower() == "ets":
            params = ETSDetectorParams()
            params.from_dict(self.algo_params)
            self.algo_params = params

        else:
            raise RuntimeError()

        return


class AnomalyDetector:
    def __init__(self, config: AnomalyDetectionConfig):
        """
        Init the anomaly detector with proper configuration. If no config provided, use default
        :param config: AnomalyDetectionConfig
        """

        if config.algo_name.lower() == "one_class_svm":
            self.anomaly_detector = OneClassSVMDetector(
                config.algo_params if config.algo_params else OneClassSVMParams()
            )

        elif config.algo_name.lower() == "isolation_forest":
            self.anomaly_detector = IsolationForestDetector(
                config.algo_params if config.algo_params else IsolationForestParams()
            )

        elif config.algo_name.lower() == "lof":
            self.anomaly_detector = LOFDetector(
                config.algo_params if config.algo_params else LOFParams()
            )

        elif config.algo_name.lower() == "distribution_divergence":
            self.anomaly_detector = DistributionDivergence(
                config.algo_params
                if config.algo_params
                else DistributionDivergenceParams()
            )

        elif config.algo_name.lower() == "dbl":
            self.anomaly_detector = DBLDetector(
                config.algo_params
                if config.algo_params
                else DBLDetectorParams()
            )

        elif config.algo_name.lower() == "ets":
            self.anomaly_detector = ETSDetector(
                config.algo_params
                if config.algo_params
                else ETSDetectorParams()
            )

        else:
            raise RuntimeError(
                "Anomaly detection algorithm {} is not defined".format(config.algo_name)
            )

        return

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
