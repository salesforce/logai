#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from logai.algorithms.anomaly_detection_algo.isolation_forest import IsolationForestDetector
from logai.algorithms.anomaly_detection_algo.local_outlier_factor import LOFDetector
from logai.algorithms.anomaly_detection_algo.one_class_svm import OneClassSVMDetector
from logai.analysis.anomaly_detector import AnomalyDetectionConfig, AnomalyDetector
from logai.utils import constants

from tests.logai.test_utils.fixtures import log_features, log_counter_df


class TestAnomalyDetector:
    def setup(self):
        self.supported_feature_algos = ['one_class_svm', 'isolation_forest', 'lof']
        self.supported_counter_algo = ['dbl', 'ets']

    def test_creating_models_default_params_one_class_svm(self):
        config = AnomalyDetectionConfig(algo_name='one_class_svm')
        anomaly_detector = AnomalyDetector(config)
        assert isinstance(anomaly_detector, AnomalyDetector), 'Clustering analyzer creation failed'
        assert isinstance(anomaly_detector.anomaly_detector, OneClassSVMDetector), 'Model instance does not match definition'

    def test_creating_models_default_params_isolation_forest(self):
        config = AnomalyDetectionConfig(algo_name='isolation_forest')
        anomaly_detector = AnomalyDetector(config)
        assert isinstance(anomaly_detector, AnomalyDetector), 'Clustering analyzer creation failed'
        assert isinstance(anomaly_detector.anomaly_detector, IsolationForestDetector), 'Model instance does not match definition'

    def test_creating_models_default_params_one_class_svm(self):
        config = AnomalyDetectionConfig(algo_name='lof')
        anomaly_detector = AnomalyDetector(config)
        assert isinstance(anomaly_detector, AnomalyDetector), 'Clustering analyzer creation failed'
        assert isinstance(anomaly_detector.anomaly_detector, LOFDetector), 'Model instance does not match definition'

    def test_fit_predict_default(self, log_features):
        for algo in self.supported_feature_algos:
            config = AnomalyDetectionConfig(algo_name=algo)
            anomaly_detector = AnomalyDetector(config)
            anomaly_detector.fit(log_features)
            labels = anomaly_detector.predict(log_features)

            assert isinstance(labels, pd.DataFrame), 'labels are not pd.DataFrame'
            assert 'anom_score' in labels.columns, 'labels must contain anom_score column'

    def test_fit_predict_counter(self, log_counter_df):
        for algo in self.supported_counter_algo:
            log_counter_df["attribute"] = log_counter_df.drop([constants.LOG_COUNTS, constants.LOG_TIMESTAMPS], axis=1).apply(
                lambda x: "-".join(x.astype(str)), axis=1)

            attr_list = log_counter_df["attribute"].unique()
            res = pd.Series()
            for attr in attr_list:
                temp_df = log_counter_df[log_counter_df["attribute"] == attr]
                if temp_df.shape[0] < constants.MIN_TS_LENGTH:
                    anom_score = np.repeat(0.0, temp_df.shape[0])
                    res = res.append(pd.Series(anom_score, index=temp_df.index))
                else:
                    train, test = train_test_split(
                        temp_df[[constants.LOG_TIMESTAMPS, constants.LOG_COUNTS]],
                        shuffle=False,
                        train_size=0.3
                    )

                    config = AnomalyDetectionConfig(algo_name=algo)
                    anomaly_detector = AnomalyDetector(config)

                    anomaly_detector.fit(train)
                    labels = anomaly_detector.predict(test)

            assert isinstance(labels, pd.DataFrame), 'labels are not pd.DataFrame'
            assert 'anom_score' in labels.columns, 'labels must contain anom_score column'
