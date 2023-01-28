#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd

from logai.algorithms.categorical_encoding_algo.label_encoding import LabelEncoding
from logai.algorithms.categorical_encoding_algo.one_hot_encoding import OneHotEncoding
from logai.algorithms.categorical_encoding_algo.ordinal_encoding import OrdinalEncoding
from logai.information_extraction.categorical_encoder import CategoricalEncoderConfig, CategoricalEncoder


class TestCategoricalEncoder:
    def test_model_creation(self):
        config = CategoricalEncoderConfig(name="label_encoder")

        encoder = CategoricalEncoder(config)
        assert isinstance(encoder.encoder, LabelEncoding), "should be label encoder"

        config = CategoricalEncoderConfig(name="one_hot_encoder")

        encoder = CategoricalEncoder(config)
        assert isinstance(encoder.encoder, OneHotEncoding), "should be label encoder"

        config = CategoricalEncoderConfig(name="ordinal_encoder")
        encoder = CategoricalEncoder(config)
        assert isinstance(encoder.encoder, OrdinalEncoding), "should be label encoder"








