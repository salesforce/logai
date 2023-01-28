#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import abc
import attr


class Config(abc.ABC):
    @classmethod
    def from_dict(cls, config_dict):
        """
        Loads a config from a config dict.

        :param config_dict: The config parameters in a dict.
        """
        if config_dict is None:
            config_dict = {}

        config = cls()
        for field in config.__dict__:
            # If there is a default and the value of the field is none we can assign a value
            if field in config_dict:
                setattr(config, field, config_dict[field])
        return config

    def as_dict(self):
        return attr.asdict(self)
