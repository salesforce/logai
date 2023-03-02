#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import random

import pandas as pd
import pytest

from logai.algorithms.nn_model.transformers import TransformerAlgo, TransformerAlgoConfig
from logai.utils import constants
from tests.logai.test_utils.fixtures import log_pattern


class TestTransformers:

    @pytest.mark.skip(reason="time consuming, skip in generally")
    def test_train_pred(self, log_pattern):
        log_pattern = log_pattern.head(10)

        labels = [0, 1]

        log_labels = pd.Series(
            [random.choice(labels) for _ in range(len(log_pattern.index))],
            name='label',
            index=log_pattern.index
        )

        config = TransformerAlgoConfig()

        transformer = TransformerAlgo(config)
        transformer.train(log_pattern, log_labels)

        y_label_pred, _, _ = transformer.predict(log_pattern, log_labels)

        for l in y_label_pred:
            assert l in [0, 1], 'result should be either 0 or 1'
