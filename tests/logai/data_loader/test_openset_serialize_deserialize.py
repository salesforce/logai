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
from logai.utils import constants

TEST_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")


class TestSerializeDeserialize:
    def setup(self):
        pass

    def test_bgl_serialize_deserialize(self):

        filepath = os.path.join(TEST_DATA_PATH, "BGL_AD/BGL_11k.log")
        output_filepath = os.path.join(TEST_DATA_PATH, "BGL_AD/BGL_11k_logrecord.csv")

        data_loader_config_dict = {
            "filepath": filepath,
            "reader_args": {
                "log_format": "<Label> <Id> <Date> <Code1> <Time> <Code2> <Content>"
            },
            "log_type": "log",
            "dimensions": {
                "body": ["Content"],
                "timestamp": ["Time"],
                "labels": ["Label"],
                "span_id": ["Id"],
            },
            "datetime_format": "%Y-%m-%d-%H.%M.%S.%f",
        }

        data_loader_config = DataLoaderConfig.from_dict(data_loader_config_dict)
        self.data_loader = FileDataLoader(data_loader_config)
        self.logrecord = self.data_loader.load_data()
        self.logrecord.save_to_csv(output_filepath)

        self.logrecord = LogRecordObject.load_from_csv(output_filepath)

    def test_hdfs_serialize_deserialize(self):
        filepath = os.path.join(TEST_DATA_PATH, "HDFS_AD/HDFS_5k.log")
        output_filepath = os.path.join(TEST_DATA_PATH, "HDFS_AD/HDFS_5k_logrecord.csv")

        data_loader_config_dict = {
            "filepath": filepath,
            "reader_args": {
                "log_format": "<Date> <Time> <Pid> <Level> <Component> <Content>"
            },
            "log_type": "log",
            "dimensions": {"body": ["Content"], "timestamp": ["Date", "Time"]},
            "datetime_format": "%y%m%d %H%M%S",
        }

        print("data_loader_config_dict ", data_loader_config_dict)
        data_loader_config = DataLoaderConfig.from_dict(data_loader_config_dict)
        self.data_loader = FileDataLoader(data_loader_config)
        self.logrecord = self.data_loader.load_data()
        self.logrecord.save_to_csv(output_filepath)

        self.logrecord = LogRecordObject.load_from_csv(output_filepath)
