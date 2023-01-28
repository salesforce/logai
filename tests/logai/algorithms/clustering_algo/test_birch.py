#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd
import pytest
from sklearn.cluster import Birch

from logai.algorithms.clustering_algo.birch import BirchParams, BirchAlgo
from tests.logai.test_utils.fixtures import empty_feature, log_features


class TestBirchAlgo:
    def setup(self):
        pass

    def test_fit_none_input(self, empty_feature):
        params = BirchParams()
        detector = BirchAlgo(params)
        assert isinstance(params, BirchParams), "params must be BirchParams"
        assert isinstance(detector, BirchAlgo), "detector must be BirchAlgo"
        with pytest.raises(ValueError):
            assert detector.fit(empty_feature)

    def test_fit_predict(self, log_features):
        params = BirchParams()
        detector = BirchAlgo(params)
        assert isinstance(params, BirchParams), "params must be BirchParams"
        assert isinstance(detector, BirchAlgo), "detector must be BirchAlgo"
        detector.fit(log_features)
        assert isinstance(detector.model, Birch), "Model must be Birch"
        res = detector.predict(log_features)
        assert isinstance(res, pd.Series), "result must be pd.Series"
