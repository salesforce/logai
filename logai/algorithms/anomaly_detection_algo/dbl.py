#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from datetime import datetime

import pandas as pd
from attr import dataclass
from typing import Tuple, List

from merlion.models.anomaly.dbl import DynamicBaseline, DynamicBaselineConfig

from logai.algorithms.algo_interfaces import AnomalyDetectionAlgo
from logai.config_interfaces import Config
from logai.utils import constants
from logai.utils.functions import pd_to_timeseries
from logai.algorithms.factory import factory


@dataclass
class DBLDetectorParams(Config):
    """
    Dynamic Baseline Parameters. For more details on the paramaters see 
    https://opensource.salesforce.com/Merlion/v1.3.1/merlion.models.anomaly.html#module-merlion.models.anomaly.dbl.

    :param threshold: The rule to use for thresholding anomaly scores.
    :param fixed_period: ``(t0, tf)``; Train the model on all datapoints occurring between t0 and tf (inclusive).
    :param train_window: A string representing a duration of time to serve as the scope for a
        rolling dynamic baseline model.
    :param wind_sz: The window size in minutes to bucket times of day. This parameter only applied
        if a daily trend is one of the trends used.
    :param trends: The list of trends to use. Supported trends are “daily”, “weekly” and “monthly”.
    """

    threshold: float = 0.0
    fixed_period: Tuple[str, str] = None
    train_window: str = None
    wind_sz: str = "1h"
    trends: List[str] = None
    kwargs: dict = {}


@factory.register("detection", "dbl", DBLDetectorParams)
class DBLDetector(AnomalyDetectionAlgo):
    """Dynamic baseline based time series anomaly detection. This is a wrapper class for the Dynamic Baseline
    anomaly detection model from Merlion library .
    https://opensource.salesforce.com/Merlion/v1.3.1/merlion.models.anomaly.html#module-merlion.models.anomaly.dbl
    Current implementation only supports anomaly detection on the constants.LOGLINE_COUNTS class (which maintains
    frequency counts of the log events).
    """
    def __init__(self, params: DBLDetectorParams):
        dbl_config = DynamicBaselineConfig(
            fixed_period=params.fixed_period,
            train_window=params.train_window,
            wind_sz=params.wind_sz,
            trends=params.trends,
            **params.kwargs
        )
        self.model = DynamicBaseline(dbl_config)
        self.min_ts_length = 10
        self.threshold = params.threshold

    def fit(self, log_features: pd.DataFrame):
        """
        Training method of the Dynamic Baseline model.

        :param log_features: A log feature dataframe that must only contain two columns
            ['timestamp': datetime, constants.LOGLINE_COUNTS: int].
        """
        self._is_valid_ts_df(log_features)

        time_series = pd_to_timeseries(log_features)
        self.model.train(time_series)

    def predict(self, log_features: pd.DataFrame):
        """
        Predicts anomaly scores for log_feature["timestamp", constants.LOGLINE_COUNTS].

        :param log_features: A log feature dataframe that must contain two columns
            ['timestamp': datetime, 'counts': int].
        :return: A dataframe of the predicted anomaly scores, e.g., index:log_features.index.
            value: anomaly score to indicate if anomaly or not.
        """
        self._is_valid_ts_df(log_features)

        index = log_features.index
        time_series = pd_to_timeseries(log_features)
        test_pred = self.model.get_anomaly_label(time_series)
        anom_score = test_pred.to_pd()
        anom_score["trainval"] = False
        anom_score.index = index
        return anom_score

    @staticmethod
    def _is_valid_ts_df(log_feature):
        columns = log_feature.columns.values

        for c in columns:
            if c not in [constants.LOG_TIMESTAMPS, constants.LOG_COUNTS]:
                raise ValueError(
                    "log feature dataframe must only contain two columns ['{}': datetime, '{}': int]".format(
                        constants.LOG_TIMESTAMPS, constants.LOG_COUNTS
                    )
                    + "Current columns: {}".format(columns)
                )

        if constants.LOG_TIMESTAMPS not in columns:
            raise ValueError(
                "dataframe must contain {} column".format(constants.LOG_TIMESTAMPS)
            )

        if constants.LOG_COUNTS not in columns:
            raise ValueError(
                "dataframe must contain {} column".format(constants.LOG_COUNTS)
            )

        for ts in log_feature[constants.LOG_TIMESTAMPS]:
            if not isinstance(ts, datetime):
                raise ValueError("{} must be datetime".format(constants.LOG_TIMESTAMPS))
