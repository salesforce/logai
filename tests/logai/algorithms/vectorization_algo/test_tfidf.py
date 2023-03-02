#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd

from logai.algorithms.vectorization_algo.tfidf import TfIdfParams, TfIdf

from tests.logai.test_utils.fixtures import logrecord_body


class TestTfIdfParams:
    def test_create_default_params(self):
        params = TfIdfParams()
        assert params.use_idf, "default to use_idf True"

    def test_set_param_values(self):
        params = TfIdfParams(use_idf=False)
        assert not params.use_idf, "set to use_idf False"


class TestTfIdf:

    def setup(self):
        pass

    def test_fit_predict(self, logrecord_body):
        params = TfIdfParams()
        model = TfIdf(params)
        assert isinstance(model, TfIdf), "not a TFIDF model"
        loglines = logrecord_body['logline']
        model.fit(loglines)
        res = model.transform(loglines)
        assert isinstance(res, pd.Series), "result is not a pandas.Series"
        assert len(loglines) == len(res), "result length should match input"
