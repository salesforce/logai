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
from logai.dataloader.data_loader import FileDataLoader, DataLoaderConfig
from logai.preprocess.bgl_preprocessor import BGLPreprocessor
from logai.preprocess.openset_preprocessor import OpenSetPreprocessor, PreprocessorConfig
from logai.utils import constants

TEST_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "test_data",
    "BGL_AD",
    "BGL_11k_logrecord.csv",
)

TEST_OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "test_data",
    "BGL_AD",
    "BGL_11k_preprocessed_logrecord.csv",
)

class TestBGLPreprocessor:
    def setup(self):
        self.logrecord = LogRecordObject.load_from_csv(TEST_DATA_PATH)
        pass

    def test_bgl_preprocess(self):

        bgl_preprocessor_config = {
            "custom_delimiters_regex": [":", ",", "=", "\t"],
            "custom_replace_list": [
                [r"(0x)[0-9a-zA-Z]+", " HEX "],
                [r"((?![A-Za-z]{8}|\d{8})[A-Za-z\d]{8})", " ALPHANUM "],
                [r"\d+.\d+.\d+.\d+", " IP "],
                [r"\d+", " INT "],
            ],
        }

        preprocessor_config = PreprocessorConfig.from_dict(bgl_preprocessor_config)
        preprocessor = BGLPreprocessor(preprocessor_config)

        clean_logrecord = preprocessor.clean_log(self.logrecord)

        clean_logrecord.save_to_csv(TEST_OUTPUT_PATH)
        
        assert " HEX " in clean_logrecord.body
        assert " ALPHANUM " in clean_logrecord.body
        assert " IP " in clean_logrecord.body
