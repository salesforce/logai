#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd
from attr import dataclass
from datetime import datetime

from logai.algorithms.algo_interfaces import AnomalyDetectionAlgo
from logai.config_interfaces import Config
from logai.utils import constants

from merlion.models.anomaly.forecast_based.ets import ETSDetector as MerlionETSDetector, ETSDetectorConfig

from logai.utils.functions import pd_to_timeseries
from logai.algorithms.factory import factory


@dataclass
class ETSDetectorParams(Config):
    """
    ETS Anomaly Detector Parameters
    """
    max_forecast_steps: int = None
    target_seq_index: int = None
    error: str = "add"
    trend: str = "add"
    damped_trend: bool = True
    seasonal: str = "add"
    seasonal_periods: str = None
    refit: bool = True
    kwargs: dict = {}


@factory.register("detection", "ets", ETSDetectorParams)
class ETSDetector(AnomalyDetectionAlgo):
    """
    ETS Anomaly Detector
    """
    def __init__(self, params: ETSDetectorParams):
        ets_config = ETSDetectorConfig(
            max_forecast_steps=params.max_forecast_steps,
            target_seq_index=params.target_seq_index,
            error=params.error,
            trend=params.trend,
            damped_trend=params.damped_trend,
            seasonal=params.seasonal,
            seasonal_periods=params.seasonal_periods,
            refit=params.refit,
            **params.kwargs
        )

        self.model = MerlionETSDetector(ets_config)

    def fit(self, log_features: pd.DataFrame):
        """
        Train
        :param log_features: log feature dataframe must only contain two columns ['timestamp': datetime, constants.LOGLINE_COUNTS: int].
        :return: train_scores: anomaly scores dataframe
        ['index':log_features.index, 'timestamps': datetime, 'anom_score': scores, 'trainval': whether it is training set.
        """
        self._is_valid_ts_df(log_features)
        index = log_features.index
        time_series = pd_to_timeseries(log_features)
        train_scores = self.model.train(time_series).to_pd()

        # ETS interpolates missing timestamps, we need to drop those here
        train_scores = train_scores.loc[time_series.to_pd().index]
        train_scores[constants.LOG_TIMESTAMPS] = train_scores.index
        train_scores['trainval'] = True
        train_scores.index = index
        return train_scores

    def predict(self, log_features: pd.DataFrame):
        """
        Predict anomaly scores for log_feature["timestamp", constants.LOGLINE_COUNTS]
        :param log_features: log feature dataframe must only contain two columns ['timestamp': datetime, constants.LOGLINE_COUNTS: int].
        :return: test_scores: anomaly scores dataframe
        ['index':log_features.index, 'timestamps': datetime, 'anom_score': scores, 'trainval': whether it is training set.
        """
        self._is_valid_ts_df(log_features)
        index = log_features.index
        time_series = pd_to_timeseries(log_features)
        test_scores = self.model.get_anomaly_label(time_series).to_pd()
        # ETS interpolates missing timestamps, we need to drop those here
        test_scores = test_scores.loc[time_series.to_pd().index]
        test_scores[constants.LOG_TIMESTAMPS] = test_scores.index
        test_scores['trainval'] = False
        test_scores.index = index
        return test_scores

    @staticmethod
    def _is_valid_ts_df(log_feature):
        columns = log_feature.columns.values

        for c in columns:
            if c not in [constants.LOG_TIMESTAMPS, constants.LOG_COUNTS]:
                raise ValueError(
                    "log feature dataframe must only contain two columns ['{}': datetime, '{}': int]".format(
                        constants.LOG_TIMESTAMPS,
                        constants.LOG_COUNTS
                    ) + "Current columns: {}".format(columns)
                )

        if constants.LOG_TIMESTAMPS not in columns:
            raise ValueError("dataframe must contain {} column".format(constants.LOG_TIMESTAMPS))

        if constants.LOG_COUNTS not in columns:
            raise ValueError("dataframe must contain {} column".format(constants.LOG_COUNTS))

        for ts in log_feature[constants.LOG_TIMESTAMPS]:
            if not isinstance(ts, datetime):
                raise ValueError("{} must be datetime".format(constants.LOG_TIMESTAMPS))
