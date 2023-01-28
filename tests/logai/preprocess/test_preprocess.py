#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd

from logai.preprocess.preprocessor import Preprocessor, PreprocessorConfig
from tests.logai.test_utils.fixtures import logrecord_body


class TestPreprocessorConfig:
    def test_from_dict(self):
        config_dict = {
            "custom_delimiters_regex": [r"`+|\s+"]
        }

        preprocess_config = PreprocessorConfig.from_dict(config_dict)
        assert preprocess_config.custom_delimiters_regex[0] == r"`+|\s+", "Config read failed"


class TestPreprocess:
    def test_clean_log(self, logrecord_body):
        loglines = logrecord_body["logline"]
        config = PreprocessorConfig(
            custom_delimiters_regex=[r"\|"],
            custom_replace_list=[[r'Step_\w+', '<Operations>']]
        )

        preprocessor = Preprocessor(config)
        clean_log, pattern_list = preprocessor.clean_log(loglines)
        # print(clean_log.head(5))
        for l in clean_log:
            assert "|" not in l, "custom delimiters not removed"
            assert "LSC_" not in l, "custom regex terms not removed"

        assert '<Operations>' in pattern_list.columns
