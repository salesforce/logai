#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import os

import pandas as pd
import pytest
from logai.dataloader.data_model import LogRecordObject
from logai.utils import constants

TEST_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")


@pytest.fixture
def logrecord_body():
    path = os.path.join(TEST_DATA_PATH, "default_logrecord_body")
    feature_set = pd.read_pickle(path)
    return feature_set


@pytest.fixture
def hdfs_logrecord():
    path = os.path.join(TEST_DATA_PATH, "HDFS_AD", "HDFS_5k_logrecord.csv")
    return LogRecordObject.load_from_csv(path)


@pytest.fixture
def bgl_logrecord():
    path = os.path.join(TEST_DATA_PATH, "BGL_AD", "BGL_11k_logrecord.csv")
    return LogRecordObject.load_from_csv(path)

@pytest.fixture
def hdfs_preprocessed_logrecord():
    path = os.path.join(TEST_DATA_PATH, "HDFS_AD", "HDFS_5k_preprocessed_logrecord.csv")
    return LogRecordObject.load_from_csv(path)

@pytest.fixture
def bgl_preprocessed_logrecord():
    path = os.path.join(TEST_DATA_PATH, "BGL_AD", "BGL_11k_preprocessed_logrecord.csv")
    return LogRecordObject.load_from_csv(path)

@pytest.fixture
def empty_feature():
    feature_set = pd.DataFrame()
    return feature_set


@pytest.fixture
def log_features():
    path = os.path.join(TEST_DATA_PATH, "default_feature_set")
    feature_set = pd.read_pickle(path)
    return feature_set


@pytest.fixture
def log_counter_df():
    path = os.path.join(TEST_DATA_PATH, "default_counter_df.csv")
    # counter_df = pd.read_pickle(path)
    counter_df = pd.read_csv(path)
    counter_df.columns = [
        "Action",
        "ID",
        constants.LOG_TIMESTAMPS,
        constants.LOG_COUNTS,
    ]
    counter_df[constants.LOG_TIMESTAMPS] = \
        pd.to_datetime(counter_df[constants.LOG_TIMESTAMPS])
    return counter_df


@pytest.fixture
def log_attributes():
    path = os.path.join(TEST_DATA_PATH, "healthapp_attributes")
    attributes = pd.read_pickle(path)
    return attributes


@pytest.fixture
def log_timestamps():
    path = os.path.join(TEST_DATA_PATH, "healthapp_timestamp")
    attributes = pd.read_pickle(path)
    return attributes


@pytest.fixture
def log_vector_w2v():
    path = os.path.join(TEST_DATA_PATH, "healthapp_log_vector_w2v")
    vector = pd.read_pickle(path)
    return vector


@pytest.fixture
def log_pattern():
    path = os.path.join(TEST_DATA_PATH, "healthapp_log_pattern")
    pattern = pd.read_pickle(path)
    return pattern
