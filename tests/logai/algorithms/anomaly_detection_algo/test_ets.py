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

from logai.algorithms.anomaly_detection_algo.ets import ETSDetector, ETSDetectorParams
from merlion.models.anomaly.forecast_based.ets import ETSDetector as MerlionETSDetector, ETSDetectorConfig
from logai.utils import constants
from logai.utils.functions import pd_to_timeseries

from tests.logai.test_utils.fixtures import log_counter_df


class TestETSDetector:
    def setup(self):
        pass

    def test_ets_detector(self, log_counter_df):
        counter_df = log_counter_df

        ts_df = counter_df[[constants.LOG_COUNTS]]
        ts_df.index = counter_df[constants.LOG_TIMESTAMPS]
        counter_df["attribute"] = counter_df.drop([constants.LOG_COUNTS, constants.LOG_TIMESTAMPS], axis=1).apply(
            lambda x: "-".join(x.astype(str)), axis=1)

        attr_list = counter_df["attribute"].unique()

        res = pd.Series()
        num_inference = 0
        for attr in attr_list:
            temp_df = counter_df[counter_df["attribute"] == attr]

            if temp_df.shape[0] < constants.MIN_TS_LENGTH:
                anom_score = np.repeat(0.0, temp_df.shape[0])
                num_inference += temp_df.shape[0]
                res = res.append(pd.Series(anom_score, index=temp_df.index))
            else:
                train, test = train_test_split(
                    temp_df[[constants.LOG_TIMESTAMPS, constants.LOG_COUNTS]],
                    shuffle=False,
                    train_size=0.3
                )
                anom_score_training = pd.Series(np.repeat(0.0, train.shape[0]), index=train.index)
                model = ETSDetector(ETSDetectorParams())

                model.fit(train)
                anom_score = model.predict(test)
                num_inference += test.shape[0]
                res = res.append(anom_score_training)
                res = res.append(anom_score['anom_score'])
        print(len(res), counter_df.shape[0])
        assert (res.index == counter_df.index).all(), "Res.index should be identical to counter_df.index"
        assert len(res) == len(counter_df.index), "length of res should be equal to length of counter_df"
