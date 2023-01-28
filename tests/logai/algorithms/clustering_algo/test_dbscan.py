#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd
import pytest
from sklearn.cluster import DBSCAN

from logai.algorithms.clustering_algo.dbscan import DbScanAlgo, DbScanParams
from tests.logai.test_utils.fixtures import empty_feature, log_features


class TestDbScanAlgo:
    def setup(self):
        pass

    def test_fit_none_input(self, empty_feature):
        params = DbScanParams()
        detector = DbScanAlgo(params)
        assert isinstance(params, DbScanParams), "params must be DbScanParams"
        assert isinstance(detector, DbScanAlgo), "detector must be DbScanAlgo"
        with pytest.raises(ValueError):
            assert detector.fit(empty_feature)

    def test_fit_predict(self, log_features):
        params = DbScanParams()
        detector = DbScanAlgo(params)
        assert isinstance(params, DbScanParams), "params must be DbScanParams"
        assert isinstance(detector, DbScanAlgo), "detector must be DbScanAlgo"
        detector.fit(log_features)
        assert isinstance(detector.model, DBSCAN), "Model must be DBSCAN"
        res = detector.predict(log_features)
        assert isinstance(res, pd.Series), "result must be pd.Series"
