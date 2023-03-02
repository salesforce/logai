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
from logai.analysis.anomaly_detector import AnomalyDetector, AnomalyDetectionConfig

from logai.algorithms.anomaly_detection_algo.anomaly_detector_het import HetAnomalyDetector, HetAnomalyDetectionConfig
from merlion.models.anomaly.forecast_based.ets import ETSDetector as MerlionETSDetector, ETSDetectorConfig
from logai.utils import constants
from logai.utils.functions import pd_to_timeseries

from tests.logai.test_utils.fixtures import log_counter_df


class TestHetAnomalyDetector:
    def setup(self):
        pass

    def test_het_anomaly_detector(self, log_counter_df):
        counter_df = log_counter_df
        model = HetAnomalyDetector(
            HetAnomalyDetectionConfig(algo_name='ets')
        )
        res = model.fit_predict(counter_df)
        print(res.head())
