#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import json
import inspect
import importlib
from collections import OrderedDict


class ParamInfoMixin:
    def get_config_class(self, algorithm):
        assert algorithm.lower() in self.algorithms, (
            f"Unknown algorithm {algorithm}. Please select from "
            f"{list(self.algorithms.keys())}."
        )

        module_info = self.algorithms[algorithm.lower()]
        module = importlib.import_module(module_info[0])
        config_class = getattr(module, module_info[2])
        return config_class

    def get_parameter_info(self, algorithm):
        config_class = self.get_config_class(algorithm)
        annotations = config_class.__annotations__

        param_info = OrderedDict()
        valid_types = [int, float, str, bool, list, tuple, dict]
        if not hasattr(config_class, "__attrs_attrs__"):
            members = inspect.getmembers(
                config_class, lambda a: not inspect.isroutine(a)
            )
            members = [m for m in members if not m[0].startswith("_")]
        else:
            members = [
                (attr.name, attr.default) for attr in config_class.__attrs_attrs__
            ]

        for name, value in members:
            if name.lower() == "verbose":
                continue
            value_type = annotations.get(name, type(value))
            if value_type in valid_types:
                param_info[name] = {"type": value_type, "default": value}
        return param_info

    def parse_parameters(self, param_info, params):
        for key in params.keys():
            assert key in param_info, f"{key} is not in `param_info`."

        kwargs = {}
        for name, value in params.items():
            info = param_info[name]
            value_type = info["type"]
            if value.lower() in ["none", "null"]:
                kwargs[name] = None
            elif value_type in [int, float, str]:
                kwargs[name] = value_type(value)
            elif value_type == bool:
                assert value.lower() in [
                    "true",
                    "false",
                ], f"The value of {name} should be either True or False."
                kwargs[name] = value.lower() == "true"
            elif info["type"] in [list, tuple, dict]:
                value = value.replace(" ", "").replace("\t", "")
                value = value.replace("(", "[").replace(")", "]").replace(",]", "]")
                kwargs[name] = json.loads(value)
        return kwargs
