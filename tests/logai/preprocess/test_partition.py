#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import os

import pandas as pd

from logai.dataloader.openset_data_loader import OpenSetDataLoader, OpenSetDataLoaderConfig
from logai.preprocess.partitioner import PartitionerConfig, Partitioner
from logai.utils import constants

TEST_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_data')


class TestPartitioner:

    def setup(self):
        dataset_name = "HDFS"

        filepath = os.path.join(TEST_DATA_PATH, "HDFS/HDFS_2000.log")

        data_loader = OpenSetDataLoader(
            OpenSetDataLoaderConfig(dataset_name, filepath)
        )

        self.logrecord = data_loader.load_data()

        pass

    def test_group_counter(self):
        logrecord = self.logrecord
        config = PartitionerConfig(
            group_by_time="15s",
            group_by_category=["Level"]
        )

        partitioner = Partitioner(config)

        partitioned_log_df = partitioner.group_counter(logrecord.to_dataframe())

        print(partitioned_log_df.head(5))
        return

    def test_group_sliding_window_sequence_group_by_time_and_categories(self):
        logrecord = self.logrecord
        config = PartitionerConfig(
            sliding_window=2,
            group_by_time="15s",
            group_by_category=["Level"]
        )

        partitioner = Partitioner(config)

        partitioned_log_df = partitioner.group_sliding_window(logrecord.to_dataframe(), constants.LOGLINE_NAME)

        assert constants.LOG_TIMESTAMPS in partitioned_log_df.columns, "{} should be a column".format(constants.LOG_TIMESTAMPS)
        assert constants.LOGLINE_NAME in partitioned_log_df.columns, "{} should be a column".format(constants.LOGLINE_NAME)
        for cat in config.group_by_category:
            assert cat in partitioned_log_df.columns, "Attribute: {} should be a column".format(cat)

        pass

    def test_group_sliding_window_sequence_group_by_categories(self):
        logrecord = self.logrecord
        config = PartitionerConfig(
            sliding_window=2,
            group_by_category=["Level"]
        )

        partitioner = Partitioner(config)

        partitioned_log_df = partitioner.group_sliding_window(logrecord.to_dataframe(), constants.LOGLINE_NAME)

        assert constants.LOGLINE_NAME in partitioned_log_df.columns, "{} should be a column".format(constants.LOGLINE_NAME)
        for cat in config.group_by_category:
            assert cat in partitioned_log_df.columns, "Attribute: {} should be a column".format(cat)

        pass

    def test_group_sliding_window_sequence_group_by_time(self):
        logrecord = self.logrecord
        config = PartitionerConfig(
            sliding_window=2,
            group_by_time="15s"
        )

        partitioner = Partitioner(config)

        partitioned_log_df = partitioner.group_sliding_window(logrecord.to_dataframe(), constants.LOGLINE_NAME)

        assert constants.LOG_TIMESTAMPS in partitioned_log_df.columns, "{} should be a column".format(constants.LOG_TIMESTAMPS)
        assert constants.LOGLINE_NAME in partitioned_log_df.columns, "{} should be a column".format(constants.LOGLINE_NAME)

        pass

    def test_group_sliding_window_sequence_no_grouping(self):
        logrecord = self.logrecord
        config = PartitionerConfig(
            sliding_window=2
        )

        partitioner = Partitioner(config)

        partitioned_log_df = partitioner.group_sliding_window(logrecord.to_dataframe(), constants.LOGLINE_NAME)

        assert constants.LOG_TIMESTAMPS not in partitioned_log_df.columns, "{} should be a column".format(
            constants.LOG_TIMESTAMPS)
        assert constants.LOGLINE_NAME in partitioned_log_df.columns, "{} should be a column".format(
            constants.LOGLINE_NAME)

    def test_sliding_window(self):
        logrecord = self.logrecord
        config = PartitionerConfig(
            sliding_window=5
        )

        partitioner = Partitioner(config)

        partitioned_loglines = partitioner.sliding_window(logrecord.body[constants.LOGLINE_NAME])

        print(partitioned_loglines.head(5).tolist())
        assert isinstance(partitioned_loglines, pd.Series), "result should be pandas.Series"
        assert partitioned_loglines.size == 2001, "length of list should be 2001"
        return



