#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import os

from logai.dataloader.data_loader import DataLoaderConfig, FileDataLoader
from logai.utils import constants

TEST_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_data')


class TestLogRecordObject:
    def setup(self):
        pass

    def test_to_dataframe(self):
        test_fpath = os.path.join(TEST_DATA_PATH, "HDFS/HDFS_2000.log")
        file_config = DataLoaderConfig(
            filepath=test_fpath,
            log_type="log",
            dimensions={
              "timestamp": ["Date", "Time"],
              "body": ["Content"],
              "attributes": ["Level"],
              "span_id": ["Pid"],
            },
            reader_args={
                "log_format": "<Date> <Time> <Pid> <Level> <Content>"
            }
        )

        dataloader = FileDataLoader(file_config)
        logrecord = dataloader.load_data()

        log_df = logrecord.to_dataframe()

        for c in log_df.columns:
            assert c in [constants.LOGLINE_NAME, constants.LOG_TIMESTAMPS, 'Level', 'span_id'], "unexpected column name."
        return