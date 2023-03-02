#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import os

from logai.dataloader.data_loader import DataLoaderConfig, FileDataLoader
from logai.dataloader.data_model import LogRecordObject
from logai.utils import constants
import pandas as pd
import json
import numpy as np
from tests.logai.test_utils.fixtures import bgl_logrecord, hdfs_logrecord

TEST_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")
TEST_OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "temp_output"
)


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
            reader_args={"log_format": "<Date> <Time> <Pid> <Level> <Content>"},
        )

        dataloader = FileDataLoader(file_config)
        logrecord = dataloader.load_data()

        log_df = logrecord.to_dataframe()

        for c in log_df.columns:
            assert c in [
                constants.LOGLINE_NAME,
                constants.LOG_TIMESTAMPS,
                "Level",
                "span_id",
            ], "unexpected column name."

    def test_serialize_deserialize(self):
        if not os.path.exists(TEST_OUTPUT_PATH):
            os.makedirs(TEST_OUTPUT_PATH)

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
            reader_args={"log_format": "<Date> <Time> <Pid> <Level> <Content>"},
        )

        dataloader = FileDataLoader(file_config)
        logrecord = dataloader.load_data()

        output_file = os.path.join(TEST_DATA_PATH, "tmp_hdfs_2000.csv")

        logrecord.save_to_csv(filepath=output_file)

        logrecord2 = LogRecordObject.load_from_csv(filepath=output_file)

        for field in logrecord2.__dataclass_fields__:
            if field == "_index":
                continue
            field_content2 = getattr(logrecord2, field).astype(str)
            field_content = getattr(logrecord, field).astype(str)
            if field_content2 is not None and field_content is not None:
                assert field_content2.equals(field_content), (
                    "field " + field + " does not match"
                )

        os.remove(output_file)

    def test_dropna(self, hdfs_logrecord):
        logrecord = hdfs_logrecord
        null_indices = set(np.random.randint(len(logrecord.body), size=5))
        logrecord.body[constants.LOGLINE_NAME][null_indices] = None
        logrecord_notnull = logrecord.dropna()
        any_null = dict(logrecord_notnull.body.isnull().any()).values()
        assert True not in any_null

    def test_select_by_index(self, bgl_logrecord):
        logrecord = bgl_logrecord
        indices_to_select = set(np.random.randint(len(logrecord.body), size=50))
        body_to_select = logrecord.body[constants.LOGLINE_NAME][indices_to_select]
        ids_to_select = logrecord.span_id[constants.SPAN_ID][indices_to_select]
        logrecord = logrecord.select_by_index(indices=indices_to_select, inplace=True)
        selected_body = logrecord.body[constants.LOGLINE_NAME]
        selected_ids = logrecord.span_id[constants.SPAN_ID]
        assert body_to_select.eq(selected_body).all(), "logrecord bodies do not match"
        assert ids_to_select.eq(selected_ids).all(), "logrecord ids do not match"

    def test_filter_by_index(self, bgl_logrecord):
        logrecord = bgl_logrecord
        len_data = len(logrecord.body)
        indices_to_filter = set(np.random.randint(len_data, size=1000))
        indices_to_select = list(set(range(len_data)) - indices_to_filter)
        body_to_filter = logrecord.body[constants.LOGLINE_NAME][indices_to_select]
        ids_to_filter = logrecord.span_id[constants.SPAN_ID][indices_to_select]
        logrecord = logrecord.filter_by_index(indices=indices_to_filter, inplace=True)
        filtered_body = logrecord.body[constants.LOGLINE_NAME]
        filtered_ids = logrecord.span_id[constants.SPAN_ID]
        assert body_to_filter.eq(filtered_body).all(), "logrecord bodies do not match"
        assert ids_to_filter.eq(filtered_ids).all(), "logrecord ids do not match"



