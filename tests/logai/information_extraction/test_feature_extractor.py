#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from logai.information_extraction.feature_extractor import FeatureExtractorConfig, FeatureExtractor, \
    _get_group_counter
from logai.utils import constants

from tests.logai.test_utils.fixtures import log_attributes, log_timestamps, log_vector_w2v, log_pattern


class TestFeatureExtractor:
    def setup(self):
        c1 = pd.Series(["A", "B", "C", "A", "C"], name="Col1")
        c2 = pd.Series(["2", "3", "1", "4", "2"], name="Col2")
        self.atr = pd.concat((c1, c2), axis=1)
        self.ts = pd.date_range(start='2015 Jul 2 10:15', end='2015 July 12', freq='12H')
        self.index = pd.Series(range(len(self.ts)))

    def test_convert_to_sequence(self, log_pattern, log_attributes, log_timestamps):
        config = FeatureExtractorConfig(
            group_by_category=["Action", "ID"],
            group_by_time="5min"
        )
        feature_extractor = FeatureExtractor(config)
        ind, seq = feature_extractor.convert_to_sequence(
            log_pattern=log_pattern,
            attributes=log_attributes,
            timestamps=log_timestamps
        )
        for x in ind['event_index']:
            assert isinstance(x, list), "index list should be list"

        for s in seq:
            assert isinstance(s, str), "log event sequence should be str"

    def test_convert_to_sequence_no_timestamps(self, log_pattern, log_attributes, log_timestamps):
        config = FeatureExtractorConfig(
            group_by_category=["Action", "ID"]
        )
        feature_extractor = FeatureExtractor(config)
        ind, seq = feature_extractor.convert_to_sequence(
            log_pattern=log_pattern,
            attributes=log_attributes
        )

        for x in ind['event_index']:
            assert isinstance(x, list), "index list should be list"

        for s in seq:
            assert isinstance(s, str), "log event sequence should be str"

    def test_covert_to_counter_vector_by_pattern(self, log_pattern):
        config = FeatureExtractorConfig(
            group_by_category=[constants.PARSED_LOGLINE_NAME]
        )
        feature_extractor = FeatureExtractor(config)
        res = feature_extractor.convert_to_counter_vector(
            log_pattern=log_pattern,
            attributes=None,
            timestamps=None
        )

        assert len(res[constants.PARSED_LOGLINE_NAME]) == len(res[constants.PARSED_LOGLINE_NAME].unique()), "Incorrect group"
        assert sum(res[constants.LOG_COUNTS]) == log_pattern.shape[0], "Counts do not sum correct"

    def test_convert_to_counter_vector_default(self, log_pattern, log_attributes, log_timestamps):
        config = FeatureExtractorConfig(
            group_by_category=["Action", "ID"]
        )
        feature_extractor = FeatureExtractor(config)
        res = feature_extractor.convert_to_counter_vector(
            log_pattern=log_pattern,
            attributes=log_attributes,
            timestamps=log_timestamps
        )
        assert len(res["Action"]) == len(res["Action"].unique()), "Incorrect group"
        assert sum(res[constants.LOG_COUNTS]) == log_attributes.shape[0], "Counts do not sum correct"

    def test_covert_to_counter_vector_no_pattern(self, log_attributes, log_timestamps):
        config = FeatureExtractorConfig(
            group_by_category=["Action", "ID"]
        )
        feature_extractor = FeatureExtractor(config)
        res = feature_extractor.convert_to_counter_vector(
            log_pattern=None,
            attributes=log_attributes,
            timestamps=log_timestamps
        )

        assert len(res["Action"]) == len(res["Action"].unique()), "Incorrect group"
        assert sum(res[constants.LOG_COUNTS]) == log_attributes.shape[0], "Counts do not sum correct"

    def test_covert_to_counter_vector_only_timestamp(self, log_pattern, log_attributes, log_timestamps):
        config = FeatureExtractorConfig(
            group_by_time="5min"
        )
        feature_extractor = FeatureExtractor(config)
        res = feature_extractor.convert_to_counter_vector(
            log_pattern=None,
            attributes=log_attributes,
            timestamps=log_timestamps
        )

        assert sum(res[constants.LOG_COUNTS]) == log_attributes.shape[0], "Counts do not sum correct"
        for ts in res[constants.LOG_TIMESTAMPS]:
            assert ts.timestamp() * 1000 % timedelta(minutes=5).total_seconds() * 1000 == 0, \
                "Timestamp not in the target frequency"

    def test_convert_to_feature_vector_default(self, log_vector_w2v, log_attributes, log_timestamps):
        config = FeatureExtractorConfig(
            group_by_time="5min",
            group_by_category=["Action", "ID"],
            max_feature_len=10
        )

        feature_extractor = FeatureExtractor(config)
        index, res = feature_extractor.convert_to_feature_vector(log_vector_w2v, log_attributes, log_timestamps)


        assert len(res.columns) == 13, "Feature size does not match"

    def test_convert_to_feature_vector_only_timestamp(self, log_vector_w2v, log_attributes, log_timestamps):
        config = FeatureExtractorConfig(
            group_by_time="5min",
            max_feature_len=10
        )

        feature_extractor = FeatureExtractor(config)
        ind, res = feature_extractor.convert_to_feature_vector(
            log_vectors=log_vector_w2v,
            attributes=None,
            timestamps=log_timestamps
        )

        assert len(res.columns) == 11, "Feature size does not match"
        for x in ind['event_index']:
            assert isinstance(x, list), "index list should be list"

    def test_convert_to_feature_vector_not_group(self, log_vector_w2v, log_attributes, log_timestamps):
        config = FeatureExtractorConfig(max_feature_len=100)

        feature_extractor = FeatureExtractor(config)
        ind, res = feature_extractor.convert_to_feature_vector(log_vector_w2v, log_attributes[['Action']], log_timestamps)

        assert len(res.columns) == 100, "Feature size does not match"

        for x in ind['event_index']:
            assert isinstance(x, int), "index list should be int if no groupby"

    def test_convert_to_feature_vector_max_len(self, log_vector_w2v, log_attributes, log_timestamps):
        config = FeatureExtractorConfig(
            max_feature_len=10
        )

        feature_extractor = FeatureExtractor(config)
        ind, res = feature_extractor.convert_to_feature_vector(log_vector_w2v, log_attributes, log_timestamps)

        assert len(res.columns) == 11, "Feature size does not match"

        assert len(ind.columns) == 14, "Index has for different domains"

        for x in ind['event_index']:
            assert isinstance(x, int), "index list should be int if no groupby"

    def test__get_group_counter(self):
        group_by_c1 = ['Col1']
        group_by_c2 = ['Col2']
        group_by_all = ['Col1', 'Col2']

        group_counter = _get_group_counter(self.atr, group_by_c1)
        assert len(group_counter.columns) == 2, "group index column does not match"
        assert len(group_counter["Col1"]) == 3, "Number of categories does not match"
        assert sum(group_counter[constants.LOG_COUNTS]) == 5, "Counts sum incorrect"

        group_counter = _get_group_counter(self.atr, group_by_c2)
        assert len(group_counter.columns) == 2, "group index column does not match"
        assert len(group_counter["Col2"]) == 4, "Number of categories does not match"
        assert sum(group_counter[constants.LOG_COUNTS]) == 5, "Counts sum incorrect"

        group_counter = _get_group_counter(self.atr, group_by_all)
        assert len(group_counter.columns) == 3, "group index column does not match"
        assert group_counter.shape[0] == 5, "Number of categories does not match"
        assert sum(group_counter[constants.LOG_COUNTS]) == 5, "Counts sum incorrect"

    def test_convert_to_feature_vector_sliding_window_time(self, log_pattern, log_attributes, log_timestamps):
        config = FeatureExtractorConfig(
            sliding_window=5,
            steps=2,
            group_by_category=["Action", "ID"],
            group_by_time="5min"
        )

        feature_extractor = FeatureExtractor(config)
        ind, res = feature_extractor.convert_to_sequence(log_pattern, log_attributes, log_timestamps)

        for x in ind['event_index']:
            assert isinstance(x, list), "index list should be list"

        for s in res:
            assert isinstance(s, str), "log event sequence should be str"

        print(res.head(3))

    def test_convert_to_feature_vector_sliding_window(self, log_pattern, log_attributes, log_timestamps):
        config = FeatureExtractorConfig(
            sliding_window=5,
            steps=2,
            group_by_category=["Action", "ID"]
        )

        feature_extractor = FeatureExtractor(config)
        ind, res = feature_extractor.convert_to_sequence(log_pattern, log_attributes, log_timestamps)

        for x in ind['event_index']:
            assert isinstance(x, list), "index list should be list"

        for s in res:
            assert isinstance(s, str), "log event sequence should be str"

        print(res.head(3))

    @pytest.mark.skip(reason="will use partitioner to convert to log sequence instead")
    def test_convert_to_sequence_sliding_window_no_grouping(self, log_pattern, log_attributes, log_timestamps):
        config = FeatureExtractorConfig(
            sliding_window=5,
            steps=2,
        )

        feature_extractor = FeatureExtractor(config)
        ind, res = feature_extractor.convert_to_sequence(log_pattern, log_attributes, log_timestamps)

        for x in ind['event_index']:
            assert isinstance(x, list), "index list should be list"

        for s in res:
            assert isinstance(s, str), "log event sequence should be str"

        print(res.head(3))


    def test_window(self, log_pattern):
        # print(log_pattern.head(5))
        for window in log_pattern.rolling(window=5):
            print(" ".join(window))





