#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import json
import os.path

from attr import dataclass

from logai.config_interfaces import Config
from logai.dataloader.data_loader import FileDataLoader, DataLoaderConfig


def get_config(dataset_name, filepath) -> DataLoaderConfig:
    """
    Retrieves the configuration of open log datasets to load data.
    
    :param dataset_name: The supported log dataset name from ("hdfs", "bgl", "HealthApp").
    :param filepath: The log file path.
    :return: The configuration to load open log datasets.
    """
    config_path = os.path.join(
        os.path.dirname(__file__),
        "openset_configs",
        "{}.json".format(dataset_name.lower()),
    )
    with open(config_path, "r") as f:
        config = DataLoaderConfig.from_dict(json.load(f))
    config.filepath = filepath
    return config


@dataclass
class OpenSetDataLoaderConfig(Config):
    dataset_name: str = None
    filepath: str = None


class OpenSetDataLoader(FileDataLoader):
    def __init__(self, config: OpenSetDataLoaderConfig):
        self._dl_config = get_config(config.dataset_name, config.filepath)
        super().__init__(self._dl_config)
        return

    def load_data(self):
        return super().load_data()

    @property
    def dl_config(self):
        return self._dl_config
