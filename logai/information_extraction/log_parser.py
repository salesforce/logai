#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import logging
import os
import pickle
from os.path import dirname, exists

import pandas as pd
import logai.algorithms.parsing_algo

from dataclasses import dataclass
from logai.config_interfaces import Config
from logai.utils import constants
from logai.algorithms.factory import factory


@dataclass
class LogParserConfig(Config):
    """
    Log Parser configuration.
    """

    parsing_algorithm: str = "drain"
    parsing_algo_params: object = None
    custom_config: object = None

    @classmethod
    def from_dict(cls, config_dict):
        config = super(LogParserConfig, cls).from_dict(config_dict)
        config.parsing_algo_params = factory.get_config(
            "parsing", config.parsing_algorithm.lower(), config.parsing_algo_params
        )
        return config


class LogParser:
    """
    Implementation of log parser for free-form text loglines.
    
    :param config: The log parser configuration.
    """

    def __init__(self, config: object):
        name = config.parsing_algorithm.lower()
        config_class = factory.get_config_class("parsing", name)
        algorithm_class = factory.get_algorithm_class("parsing", name)
        self.parser = algorithm_class(
            config.parsing_algo_params if config.parsing_algo_params else config_class()
        )

    def fit(self, loglines: pd.Series):
        """
        Trains log parser with training loglines.
        :param loglines: A pd.Series object containing the list of loglines for training.
        """
        self.parser.fit(loglines)

    def parse(self, loglines: pd.Series) -> pd.DataFrame:
        """
        Uses the trained log parser to parse loglines.
        :param loglines: A pd.Series object conatining the loglines for parsing.
        :return: A dataframe of parsed result ["loglines", "parsed_loglines", "parameter_list"].
        """
        if self.parser is None:
            raise RuntimeError("Parser is None.")
        parsed_loglines = self.parser.parse(loglines)
        if loglines.name is not constants.LOGLINE_NAME:
            loglines.name = constants.LOGLINE_NAME
        parsed_loglines.name = constants.PARSED_LOGLINE_NAME
        parsed_result = pd.concat([loglines, parsed_loglines], axis=1)

        parsed_result[constants.PARAMETER_LIST_NAME] = parsed_result.apply(
            self.get_parameter_list, axis=1
        )
        return parsed_result

    def fit_parse(self, loglines: pd.Series) -> pd.DataFrame:
        """
        Trains and parses the given loglines.
        :param loglines: A pd.Series object containing the list of loglines to train and parse.
        :return: A dataframe of parsed result ["loglines", "parsed_loglines", "parameter_list"].
        """
        try:
            self.fit(loglines)
        except RuntimeError as e:
            logging.ERROR("Cannot train parser")

        return self.parse(loglines)

    def save(self, out_path):
        """
        Saves the parser model.
        :param out_path: The directory to save parser models.
        """

        if not exists(dirname(out_path)):
            try:
                os.makedirs(dirname(out_path))
            except OSError as exc:
                if exc.errno != exc.errno.EEXIST:
                    raise RuntimeError(
                        "{} is not a valid output directory!".format(out_path)
                    )

        with open(out_path, "wb") as f:
            pickle.dump(self.parser, f)
            f.close()

    def load(self, model_path):
        """
        Loads existing parser models.
        :param model_path: The directory to load parser models.
        """

        with open(model_path, "rb") as f:
            self.parser = pickle.load(f)
            f.close()

    @staticmethod
    def get_parameter_list(row):
        """
        Returns parameter list of the loglines.

        :param row: The row in dataframe as function input containing ['logline', 'parsed_logline'].
        :return: The list of dynamic parameters.
        """
        parameter_list = []
        if not isinstance(row.logline, str) or not isinstance(row.parsed_logline, str):
            return parameter_list
        ll = row.logline.split()
        for t in ll:
            t = t.strip()
            if not t or t in row.parsed_logline:
                continue
            parameter_list.append(t)
        return parameter_list
