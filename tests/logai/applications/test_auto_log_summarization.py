#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import json
import os.path

import pandas as pd

from logai.algorithms.parsing_algo.drain import DrainParams
from logai.applications.auto_log_summarization import AutoLogSummarization
from logai.applications.application_interfaces import WorkFlowConfig
from logai.dataloader.data_loader import DataLoaderConfig
from logai.information_extraction.feature_extractor import FeatureExtractorConfig
from logai.information_extraction.log_parser import LogParserConfig
from logai.preprocess.preprocessor import PreprocessorConfig

TEST_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_data/HealthApp_2000.log')
TEST_HDFS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_data/HDFS/HDFS_2000.log')


class TestAutoLogSummarization:
    def setup(self):
        self.config = WorkFlowConfig(
            data_loader_config=DataLoaderConfig(
                filepath=TEST_LOG_PATH,
                log_type='csv',
                dimensions=dict({
                    "attributes": ["Action", "ID"],
                    "body": ["Details"]
                }),
                reader_args={
                    "header": None,
                    "sep": "|",
                    "on_bad_lines": "skip",
                    "names": ["Timestamps", "Action", "ID", "Details"]
                },
                datetime_format='%Y%m%d-%H:%M:%S:%f'
            ),
            feature_extractor_config=FeatureExtractorConfig(),
            preprocessor_config=PreprocessorConfig(
                custom_delimiters_regex=None
            ),
            log_parser_config=LogParserConfig(
                parsing_algorithm='drain',
                parsing_algo_params=DrainParams(
                    sim_th=0.4
                )
            ),
        )
        pass

    def test_find_para_list(self):
        parsing_app = AutoLogSummarization(self.config)
        parsing_app.execute()

        log_pattern = '* * 0 4'
        res = parsing_app.get_parameter_list(log_pattern)
        assert len(res['position']) == 2, "The parameter list should have 2 positions"
        return

    def test_find_log_pattern(self):
        parsing_app = AutoLogSummarization(self.config)
        parsing_app.execute()

        logline = 'onReceive action: android.intent.action.SCREEN_ON'
        log_pattern, para_list = parsing_app.find_log_pattern(logline, True)

        assert isinstance(log_pattern, str), "log pattern is an string"
        assert isinstance(para_list, pd.DataFrame), "parameter list should be a pd.DataFrame"

    def test_openset_data_loader_hdfs(self):
        json_config = """{
          "open_set_data_loader_config": {
            "dataset_name": "HDFS"
          },
          "preprocessor_config": {
              "custom_delimiters_regex":[]
          },
          "log_parser_config": {
            "parsing_algorithm": "drain",
            "parsing_algo_params": {
              "sim_th": 0.5,
              "depth": 5
            }
          }
        }
        """
        config = json.loads(json_config)
        print(config)
        workflow_config = WorkFlowConfig.from_dict(config)
        print(workflow_config)
        workflow_config.open_set_data_loader_config.filepath = TEST_HDFS_PATH

        parsing_app = AutoLogSummarization(workflow_config)
        parsing_app.execute()
