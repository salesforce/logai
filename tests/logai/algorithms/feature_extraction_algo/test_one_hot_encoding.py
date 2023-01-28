#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd

from logai.algorithms.categorical_encoding_algo.one_hot_encoding import OneHotEncodingParams, OneHotEncoding


class TestOneHotEncoding:
    def setup(self):
        feature_animal = pd.Series(['cat', 'cat', 'dog', 'bird', 'dog'], name="animals")
        feature_move = pd.Series(['jump', 'run', 'sleep', 'stand', 'fly'], name="moves")

        self.log_attributes_1d = pd.DataFrame(feature_animal)
        self.log_attributes_2d = pd.concat((feature_animal, feature_move), axis=1)

    def test_model_creation_1d_attributes(self):
        params = OneHotEncodingParams()
        vectorizer = OneHotEncoding(params)
        vectorizer.model.fit(self.log_attributes_1d)
        res = vectorizer.fit_transform(self.log_attributes_1d)

        assert isinstance(res, pd.DataFrame), "results not in pandas.DataFrame"
        assert len(res.columns) == 3, "encoding length does not match"

    def test_model_creation_2d_attributes(self):
        params = OneHotEncodingParams()
        vectorizer = OneHotEncoding(params)
        vectorizer.model.fit(self.log_attributes_2d)
        res = vectorizer.fit_transform(self.log_attributes_2d)

        assert isinstance(res, pd.DataFrame), "results not in pandas.DataFrame"
        assert len(res.columns) == 8, "encoding length does not match"
