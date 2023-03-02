#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import os
import pandas as pd

from logai.dataloader.data_loader import DataLoaderConfig, FileDataLoader
from logai.dataloader.data_model import LogRecordObject
from logai.utils import constants

TEST_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_data')


class TestFileDataLoader:
    """
    Tests for data loader
    """

    def test_read_csv(self):
        test_fpath = os.path.join(TEST_DATA_PATH, "HealthApp_format_2000.csv")
        file_config = DataLoaderConfig(
            filepath=test_fpath,
            log_type='csv',
            dimensions=dict({
                "attributes": ["Action", "ID"],
                "body": ["Details"]
            }),
            reader_args={
                "header": 0
            },
            datetime_format='%Y%m%d-%H:%M:%S:%f'
        )
        self._is_valid(file_config)

    def test_read_tsv(self):
        test_fpath = os.path.join(TEST_DATA_PATH, "HealthApp_format_2000.tsv")
        file_config = DataLoaderConfig(
            filepath=test_fpath,
            log_type='tsv',
            dimensions=dict({
                "attributes": ["Action", "ID"],
                "body": ["Details"]
            }),
            reader_args={
                "header": 0
            },
            datetime_format='%Y%m%d-%H:%M:%S:%f'
        )

        self._is_valid(file_config)

    def test_read_with_sep(self):

        test_fpath = os.path.join(TEST_DATA_PATH, "HealthApp_2000.log")
        file_config = DataLoaderConfig(
            filepath=test_fpath,
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
        )

        self._is_valid(file_config)

    def test_read_without_dimensions(self):
        test_fpath = os.path.join(TEST_DATA_PATH, "HealthApp_2000.log")
        file_config = DataLoaderConfig(
            filepath=test_fpath,
            log_type='csv',
            reader_args={
                "on_bad_lines": "skip",
            },
            datetime_format='%Y%m%d-%H:%M:%S:%f'
        )
        dataloader = FileDataLoader(file_config)
        logrecord = dataloader.load_data()
        assert logrecord.attributes.empty, "Should not have attributes"
        assert len(logrecord.body.columns) == 1, "Should only have one body column"
        assert logrecord.body.columns[0] == constants.LOGLINE_NAME, "body name does not match"

    def test_read_with_log_format(self):
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
            },
            infer_datetime=True,
            datetime_format='%y%m%d %H%M%S'
        )

        dataloader = FileDataLoader(file_config)
        logrecord = dataloader.load_data()

        assert logrecord.body.columns[0] == "logline", "Body column name is not correct."
        assert logrecord.attributes.columns[0] == 'Level', "Attributes column name is not correct."
        assert logrecord.span_id.columns[0] == 'span_id', "Span_id name is not correct."
        assert (logrecord.timestamp.columns[0]) == constants.LOG_TIMESTAMPS, "Timestamp should contain 1 columns"
        print(logrecord.timestamp.head(5))

    def test_read_with_log_format_bgl(self):
        test_fpath = os.path.join(TEST_DATA_PATH, "BGL_2000.log")
        file_config = DataLoaderConfig(
            filepath=test_fpath,
            log_type="log",
            dimensions={
                "timestamp": ["Date"],
                "body": ["Content"],
                "attributes": ["Code1", "Code2"],
                "span_id": ["Id"],
            },
            reader_args={
                "log_format": "<Label> <Id> <Date> <Code1> <Time> <Code2> <Content>"
            },
            infer_datetime=True,
            datetime_format='%Y.%m.%d'

        )

        dataloader = FileDataLoader(file_config)
        logrecord = dataloader.load_data()

        assert logrecord.body.columns[0] == "logline", "Body column name is not correct."
        assert logrecord.attributes.columns[0] == 'Code1', "Attributes column name is not correct."
        assert logrecord.span_id.columns[0] == 'span_id', "Span_id name is not correct."
        assert len(logrecord.timestamp.columns) == 1, "Timestamp should contain 1 columns"
        print(logrecord.body.head(5))

    def test_to_datetime(self):
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
            },
            datetime_format='%y%m%d %H%M%S'
        )

        dataloader = FileDataLoader(file_config)
        logrecord = dataloader.load_data()

        assert logrecord.body.columns[0] == "logline", "Body column name is not correct."
        assert logrecord.attributes.columns[0] == 'Level', "Attributes column name is not correct."
        assert logrecord.span_id.columns[0] == 'span_id', "Span_id name is not correct."
        assert len(logrecord.timestamp.columns) == 1, "Timestamp should contain 2 columns"
        print(logrecord.timestamp.head(5))

    # Script to create test files.

    def _is_valid(self, file_config):
        dataloader = FileDataLoader(file_config)
        logrecord = dataloader.load_data()

        assert len(logrecord.body.columns) == 1, "Body should be single column dataframe"
        assert logrecord.body.columns[0] == constants.LOGLINE_NAME, "Logline name does not match"

        assert len(logrecord.attributes.columns) == 2, "Attributes should contain 2 columns"
        for c in logrecord.attributes.columns:
            assert c in ["Action", "ID"], "Attribute column name does not match"
