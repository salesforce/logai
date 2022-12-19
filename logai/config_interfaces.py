#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import abc
from dataclasses import fields


class Config(abc.ABC):
    @abc.abstractmethod
    def from_dict(self, config_dict):
        if not config_dict:
            return
        for field in self.__dict__:
            # If there is a default and the value of the field is none we can assign a value
            if field in config_dict:
                setattr(self, field, config_dict[field])

        return

    def as_dict(self):
        d = {}
        for field in fields(self.__class__):
            val = getattr(self, field.name)
            d[field.name] = val
        return d
