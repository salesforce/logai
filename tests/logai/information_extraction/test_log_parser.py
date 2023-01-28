#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd

from logai.algorithms.parsing_algo.drain import DrainParams
from logai.information_extraction.log_parser import LogParser, LogParserConfig
from logai.utils import constants
from tests.logai.test_utils.fixtures import logrecord_body


class TestLogParserConfig:
    def test_from_dict(self):
        config_dict = {
            "parsing_algorithm": "drain",
            "parsing_algo_params": None
        }

        config = LogParserConfig.from_dict(config_dict)
        print(config)
        assert config.parsing_algorithm == "drain", "Algorithm is not the target one"
        # assert config.parsing_algo_params is None, "Param is not None"
        assert config.custom_config is None, "Custom config is not None"

    def test_from_dict_with_drain_params(self):
        config_dict = {
            "parsing_algorithm": "drain",
            "parsing_algo_params": {
                "sim_th": 0.2,
                "extra_delimiters": [',']
            }
        }

        config = LogParserConfig.from_dict(config_dict)
        print(config)
        assert config.parsing_algorithm == "drain", "Algorithm is not the target one"
        assert isinstance(config.parsing_algo_params, DrainParams), "Params is not instance of DrainParams"
        assert config.custom_config is None, "Custom config is not None"


class TestLogParser:
    def setup(self):
        pass

    def test_fit_and_parse(self, logrecord_body):
        config = LogParserConfig()
        parser = LogParser(config)
        assert isinstance(parser, LogParser), "Log Parser creation failed."

        loglines = logrecord_body[constants.LOGLINE_NAME]

        parser.fit(loglines)
        assert parser.parser is not None, "Parsing model are not successfully trained."
        res = parser.parse(loglines)
        assert isinstance(res, pd.DataFrame), "Parsing result format is not pd.DataFrame"

    def test_fit_parse(self, logrecord_body):
        config = LogParserConfig()
        parser = LogParser(config)
        assert isinstance(parser, LogParser), "Log Parser creation failed."

        loglines = logrecord_body[constants.LOGLINE_NAME]

        res = parser.fit_parse(loglines)

        assert parser.parser is not None, "Parsing model are not successfully trained."
        assert isinstance(res, pd.DataFrame), "Parsing result format is not pd.DataFrame"

