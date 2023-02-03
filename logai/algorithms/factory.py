#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from logai.utils.misc import is_torch_available, \
    is_transformers_available


class AlgorithmFactory:
    """
    The singleton factory class for all the supported algorithms.
    """

    _algorithms = {
        "detection": {},
        "parsing": {},
        "clustering": {},
        "vectorization": {},
    }
    _algorithms_with_torch = {
        "lstm", "cnn", "transformer",
        "logbert", "forecast_nn"
    }

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(AlgorithmFactory, cls).__new__(cls)
        return cls.instance

    @classmethod
    def register(cls, task, name, config_class):
        """
        Registers the algorithm name and the configuration class for an algorithm class.

        :param task: The task name, e.g., detection, clustering
        :param name: The algorithm name(s).
        :param config_class: The configuration class.
        """

        def wrap(algo_class):
            assert (
                task in cls._algorithms
            ), f"Unknown task {task}, please choose from {cls._algorithms.keys()}."
            names = [name] if isinstance(name, str) else name
            for algo_name in names:
                assert (
                    algo_name not in cls._algorithms[task]
                ), f"Algorithm {algo_name} has been already registered."
                cls._algorithms[task][algo_name] = (config_class, algo_class)
            return algo_class

        return wrap

    @classmethod
    def unregister(cls, task, name):
        """
        Unregisters a registered algorithm.

        :param task: The task name.
        :param name: The algorithm name.
        """
        return cls._algorithms[task].pop(name, None)

    def _check_algorithm(self, task, name):
        if name in self._algorithms_with_torch:
            if not is_torch_available() or not is_transformers_available():
                raise ImportError("Some deep learning packages are missing. "
                                  "Please install them via `pip install logai[deep-learning]`.")
        assert name in self._algorithms[task], \
            f"Unknown algorithm {name}, please choose from {self._algorithms[task].keys()}."

    def get_config_class(self, task, name):
        """
        Gets the corresponding configuration class given an algorithm name.

        :param task: The task name.
        :param name: The algorithm name.
        """
        self._check_algorithm(task, name)
        return self._algorithms[task][name][0]

    def get_algorithm_class(self, task, name):
        """
        Gets the corresponding algorithm class given an algorithm name.

        :param task: The task name.
        :param name: The algorithm name.
        """
        self._check_algorithm(task, name)
        return self._algorithms[task][name][1]

    def get_config(self, task, name, config_dict):
        """
        Gets a configuration instance given an algorithm name and a config dict.

        :param task: The task name.
        :param name: The algorithm name.
        :param config_dict: The config dictionary.
        """
        self._check_algorithm(task, name)
        return self._algorithms[task][name][0].from_dict(config_dict)

    def get_algorithm(self, task, name, config):
        """
        Gets a algorithm instance given an algorithm name and a config instance.

        :param task: The task name.
        :param name: The algorithm name.
        :param config: The config instance.
        """
        self._check_algorithm(task, name)
        config_class, algorithm_class = self._algorithms[task][name]
        if config and config.algo_params:
            assert isinstance(
                config.algo_params, config_class
            ), f"`config` must be an instance of {config_class.__name__}."
        algorithm = algorithm_class(
            config.algo_params if config and config.algo_params else config_class()
        )
        return algorithm


factory = AlgorithmFactory()
