#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import os

from logai.dataloader.openset_data_loader import OpenSetDataLoader, OpenSetDataLoaderConfig

TEST_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_data')

class TestOpenSetDataLoader:
    def setup(self):
        pass

    def test_load_hdfs_data(self):

        dataset_name = "HDFS"

        filepath = os.path.join(TEST_DATA_PATH, "HDFS/HDFS_2000.log")

        data_loader = OpenSetDataLoader(
            OpenSetDataLoaderConfig(dataset_name, filepath)
        )

        log_object = data_loader.load_data()

        print(log_object.body.head(5))
        print(log_object.attributes.head(5))
        print(log_object.timestamp.head(5))

    def test_load_bgl_data(self):
        dataset_name = "BGL"

        filepath = os.path.join(TEST_DATA_PATH, "BGL_2000.log")

        data_loader = OpenSetDataLoader(
            OpenSetDataLoaderConfig(dataset_name, filepath)
        )

        log_object = data_loader.load_data()
        print(log_object.body.head(5))
        print(log_object.attributes.head(5))
        print(log_object.timestamp.head(5))

    def test_load_healthapp_data(self):

        dataset_name = "HealthApp"

        filepath = os.path.join(TEST_DATA_PATH, "HealthApp_2000.log")

        data_loader = OpenSetDataLoader(
            OpenSetDataLoaderConfig(dataset_name, filepath)
        )

        log_object = data_loader.load_data()
        print(log_object.body.head(5))
        print(log_object.attributes.head(5))
        print(log_object.timestamp.head(5))

