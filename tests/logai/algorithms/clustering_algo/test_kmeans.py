#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd
import pytest
from sklearn.cluster import KMeans

from logai.algorithms.clustering_algo.kmeans import KMeansAlgo, KMeansParams
from tests.logai.test_utils.fixtures import empty_feature, log_features


class TestKMeansAlgo:
    def setup(self):
        pass

    def test_fit_none_input(self, empty_feature):
        params = KMeansParams()
        detector = KMeansAlgo(params)
        assert isinstance(params, KMeansParams), "params must be KMeansParams"
        assert isinstance(detector, KMeansAlgo), "detector must be KMeansAlgo"
        with pytest.raises(ValueError):
            assert detector.fit(empty_feature)

    def test_fit_predict(self, log_features):
        params = KMeansParams()
        detector = KMeansAlgo(params)
        assert isinstance(params, KMeansParams), "params must be KMeansParams"
        assert isinstance(detector, KMeansAlgo), "detector must be KMeansAlgo"
        detector.fit(log_features)
        assert isinstance(detector.model, KMeans), "Model must be KMeans"
        res = detector.predict(log_features)
        assert isinstance(res, pd.Series), "result must be pd.Series"
