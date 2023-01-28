#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from logai.preprocess.openset_partitioner import (
    OpenSetPartitioner,
    OpenSetPartitionerConfig,
)
from logai.utils import constants
from logai.preprocess.hdfs_preprocessor import HDFSPreprocessor
from logai.preprocess.bgl_preprocessor import BGLPreprocessor
from logai.preprocess.preprocessor import PreprocessorConfig
from tests.logai.test_utils.fixtures import hdfs_preprocessed_logrecord, bgl_preprocessed_logrecord
import os 

class TestOpenSetPartitioner:
    def setup(self):

        pass

    def test_hdfs_generate_sliding_window(self, hdfs_preprocessed_logrecord):
        
        config = OpenSetPartitionerConfig(sliding_window=5, session_window=False)
        partitioner = OpenSetPartitioner(config=config)
        partitioner.generate_sliding_window(hdfs_preprocessed_logrecord)

    def test_hdfs_generate_session_window(self, hdfs_preprocessed_logrecord):
        config = OpenSetPartitionerConfig(sliding_window=0, session_window=True)
        partitioner = OpenSetPartitioner(config=config)
        partitioner.generate_session_window(hdfs_preprocessed_logrecord)

    def test_bgl_generate_sliding_window(self, bgl_preprocessed_logrecord):
        config = OpenSetPartitionerConfig(sliding_window=5, session_window=False)
        partitioner = OpenSetPartitioner(config=config)
        partitioner.generate_sliding_window(bgl_preprocessed_logrecord)

    def test_bgl_generate_session_window(self, bgl_preprocessed_logrecord):
        config = OpenSetPartitionerConfig(sliding_window=0, session_window=True)
        partitioner = OpenSetPartitioner(config=config)
        partitioner.generate_session_window(bgl_preprocessed_logrecord)
