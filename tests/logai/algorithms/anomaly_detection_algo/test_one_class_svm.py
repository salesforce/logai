#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd
import pytest
from sklearn.svm import OneClassSVM
from logai.algorithms.anomaly_detection_algo.one_class_svm import OneClassSVMParams, OneClassSVMDetector
from tests.logai.test_utils.fixtures import empty_feature, log_features


class TestOneClassSVM:
    def setup(self):
        pass

    def test_fit_none_input(self, empty_feature):
        params = OneClassSVMParams()
        detector = OneClassSVMDetector(params)
        assert isinstance(params, OneClassSVMParams), "params must be OneClassSVMParams"
        assert isinstance(detector, OneClassSVMDetector), "detector must be OneCklassSVMDetector"
        with pytest.raises(ValueError):
            assert detector.fit(empty_feature)

    def test_fit_predict(self, log_features):
        params = OneClassSVMParams()
        detector = OneClassSVMDetector(params)
        assert isinstance(params, OneClassSVMParams), "params must be OneClassSVMParams"
        assert isinstance(detector, OneClassSVMDetector), "detector must be OneCklassSVMDetector"
        detector.fit(log_features)
        assert isinstance(detector.model, OneClassSVM), "Model must be OneClassSVM"
        res = detector.predict(log_features)
        assert isinstance(res, pd.DataFrame), "result must be pd.DataFrame"
