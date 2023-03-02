#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import numpy as np
import pandas as pd

from logai.algorithms.anomaly_detection_algo.dbl import DBLDetector, DBLDetectorParams
from logai.utils import constants

from tests.logai.test_utils.fixtures import log_counter_df


class TestDBLDetector:
    def setup(self):
        pass

    def test_dbl(self, log_counter_df):
        counter_df = log_counter_df

        ts_df = counter_df[[constants.LOG_COUNTS]]
        ts_df.index = counter_df[constants.LOG_TIMESTAMPS]
        counter_df["attribute"] = counter_df.drop([constants.LOG_COUNTS, constants.LOG_TIMESTAMPS], axis=1).apply(lambda x: "-".join(x.astype(str)), axis=1)

        attr_list = counter_df["attribute"].unique()

        res = pd.Series()
        for attr in attr_list:
            temp_df = counter_df[counter_df["attribute"] == attr][[constants.LOG_TIMESTAMPS, constants.LOG_COUNTS]]
            if temp_df.shape[0] < constants.MIN_TS_LENGTH:
                anom_score = np.repeat(0.0, temp_df.shape[0])
                res = res.append(pd.Series(anom_score, index=temp_df.index))
            else:
                params = DBLDetectorParams(
                    wind_sz="1min"
                )
                model = DBLDetector(params)
                model.fit(temp_df)

                anom_score = model.predict(temp_df)['anom_score']
                res = res.append(anom_score)
        assert (res.index == counter_df.index).all(), "Res.index should be identical to counter_df.index"
        assert len(res) == len(counter_df.index), "length of res should be equal to length of counter_df"
