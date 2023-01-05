#
# Copyright (c) 2022 Salesforce.com, inc.
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
from dataclasses import dataclass

import logai.algorithms.parsing_algo
from logai.algorithms.parsing_algo.drain import DrainParams
from logai.config_interfaces import Config
from logai.utils import constants
from logai.algorithms.factory import factory


@dataclass
class LogParserConfig(Config):
    """
    Log Parser configuration
    """
    parsing_algorithm: str = "drain"
    parsing_algo_params: object = None
    custom_config: object = None

    def from_dict(self, config_dict):
        super().from_dict(config_dict)
        if self.parsing_algorithm and self.parsing_algo_params:
            if self.parsing_algorithm.lower() == "drain":
                params = DrainParams()
                params.from_dict(self.parsing_algo_params)
                self.parsing_algo_params = params


class LogParser:
    """
    Implementation of log parser for free-form text loglines.
    """
    def __init__(self, config: object):
        """
        Initialization of log parser.
        :param config: LogParserConfig: log parser configuration.
        """
        name = config.parsing_algorithm.lower()
        config_class = factory.get_config_class("parsing", name)
        algorithm_class = factory.get_algorithm_class("parsing", name)
        self.parser = algorithm_class(
            config.parsing_algo_params if config.parsing_algo_params else config_class())

    def fit(self, loglines: pd.Series):
        """
        Train log parser with training loglines.
        :param loglines: pd.Series: the list of loglines for training
        :return:
        """
        self.parser.fit(loglines)

    def parse(self, loglines: pd.Series) -> pd.DataFrame:
        """
        Use the trained log parser to parse loglines
        :param loglines: pd.Series: the loglines for parsing
        :return: pd.DataFrame: a dataframe of parsed result ["loglines", "parsed_loglines", "parameter_list"]
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
        Train and parse the given loglines
        :param loglines: pd.Series: the list of loglines to train and parse
        :return: pd.DataFrame: a dataframe of parsed result ["loglines", "parsed_loglines", "parameter_list"]
        """
        try:
            self.fit(loglines)
        except RuntimeError as e:
            logging.ERROR("Cannot train parser")

        return self.parse(loglines)

    def save(self, out_path):
        """
        Save the parser model
        :param out_path: the directory to save parser models.
        :return:
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
        Load existing parser models.
        :param model_path: The directory to load parser models
        :return:
        """

        with open(model_path, "rb") as f:
            self.parser = pickle.load(f)
            f.close()

    @staticmethod
    def get_parameter_list(row):
        """
        Return parameter list of the loglines
        :param row: row in dataframe as function input containing ['logline', 'parsed_logline']
        :return: list of dynamic parameters
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
