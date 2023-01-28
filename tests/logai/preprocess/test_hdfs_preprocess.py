#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import os

import pandas as pd

from logai.dataloader.data_model import LogRecordObject
from logai.preprocess.hdfs_preprocessor import HDFSPreprocessor
from logai.preprocess.openset_preprocessor import OpenSetPreprocessor, PreprocessorConfig
from logai.utils import constants

TEST_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "test_data",
    "HDFS_AD",
    "HDFS_5k_logrecord.csv"
)
TEST_LABEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "test_data",
    "HDFS_AD",
    "anomaly_label.csv"
)
TEST_OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "test_data",
    "HDFS_AD",
    "HDFS_5k_preprocessed_logrecord.csv"
)

class TestHDFSPreprocessor:
    def setup(self):
        self.logrecord = LogRecordObject.load_from_csv(TEST_DATA_PATH)

        pass

    def test_hdfs_preprocess(self):

        hdfs_preprocessor_config = {
            "custom_delimiters_regex": [":", ",", "=", "\t"],
            "custom_replace_list": [
                [r"(blk_-?\d+)", " BLOCK "],
                [r"/?/*\d+\.\d+\.\d+\.\d+", " IP "],
                [r"(0x)[0-9a-zA-Z]+", " HEX "],
                [r"\d+", " INT "],
            ],
        }

        preprocessor_config = PreprocessorConfig.from_dict(hdfs_preprocessor_config)
        preprocessor = HDFSPreprocessor(preprocessor_config, label_file=TEST_LABEL_PATH)

        clean_logrecord = preprocessor.clean_log(self.logrecord)
        clean_logrecord.save_to_csv(TEST_OUTPUT_PATH)

        assert " BLOCK " in clean_logrecord.body
        assert " HEX " in clean_logrecord.body
        assert " IP " in clean_logrecord.body
