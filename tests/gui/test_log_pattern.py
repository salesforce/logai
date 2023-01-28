#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import os
import unittest
import pandas as pd

from gui.demo.log_pattern import LogPattern
from gui.pages.utils import create_banner
from logai.applications.application_interfaces import WorkFlowConfig, \
    DataLoaderConfig, FeatureExtractorConfig, PreprocessorConfig, \
    LogParserConfig
from logai.algorithms.parsing_algo.drain import DrainParams
from logai.dataloader.openset_data_loader import OpenSetDataLoader, OpenSetDataLoaderConfig

pd.set_option('display.max_columns', None)

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class TestLogPattern(unittest.TestCase):

    def setUp(self) -> None:

        self.config = WorkFlowConfig(
            open_set_data_loader_config=OpenSetDataLoaderConfig(
                dataset_name="HealthApp",
                filepath=os.path.join(DIR, "HealthApp_2000.log")
            ),
            feature_extractor_config=FeatureExtractorConfig(),
            preprocessor_config=PreprocessorConfig(
                custom_delimiters_regex=None
            ),
            log_parser_config=LogParserConfig(
                parsing_algorithm='drain',
                parsing_algo_params=DrainParams(
                    sim_th=0.4,
                    depth=10
                )
            ),
        )

    def test(self):
        processor = LogPattern()
        processor.execute_auto_parsing(self.config)
        patterns = processor.get_log_parsing_patterns()

        result = processor.result_table
        print(len(result.logline.unique()), len(result.parsed_logline.unique()))

        print(processor.summary_graph_df())

        count_ts = result[['timestamp', 'parsed_logline']].groupby(pd.Grouper(key='timestamp', freq='60Min', offset=0, label='right')).size().reset_index(name='count')

        print(count_ts)

        pass

    def test_ymal_to_config(self):
        yaml_config = '''
open_set_data_loader_config:
  dataset_name: 'HealthApp'
preprocessor_config:
  custom_delimiters_regex: None
    
log_parser_config:
  parsing_algorithm: 'drain'
  parsing_algo_params:
    sim_th: 0.1
    extra_delimiters: []
'''
        processor = LogPattern()
        workflow_config = processor.yaml_to_config(yaml_config)
        workflow_config.open_set_data_loader_config.filepath = os.path.join(DIR, "HealthApp_2000.log")
        processor.execute_auto_parsing(workflow_config)
        patterns = processor.get_log_parsing_patterns()

        print(workflow_config)

        pass

    def test_json_to_config(self):
        json_config = '''
{
    "open_set_data_loader_config": {
        "dataset_name": "HealthApp"
    }, 
    "preprocessor_config": {
        "custom_delimiters_regex": "None"
    }, 
    "log_parser_config": {
        "parsing_algorithm": "drain", 
        "parsing_algo_params": {
            "sim_th": 0.1, 
            "extra_delimiters": []
        }
    }
}        
        
        '''
        processor = LogPattern()
        workflow_config = processor.json_to_config(json_config)
        workflow_config.open_set_data_loader_config.filepath = os.path.join(DIR, "HealthApp_2000.log")
        processor.execute_auto_parsing(workflow_config)
        patterns = processor.get_log_parsing_patterns()
        print(workflow_config)

        pass


if __name__ == "__main__":
    unittest.main()
