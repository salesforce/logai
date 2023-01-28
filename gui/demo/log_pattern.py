#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import logging
import json
import numpy as np
import yaml
import pandas as pd

from logai.applications.auto_log_summarization import AutoLogSummarization
from logai.applications.application_interfaces import WorkFlowConfig
from .utils import ParamInfoMixin


class LogPattern(ParamInfoMixin):
    algorithms = {
        "drain": ("logai.algorithms.parsing_algo.drain", "Drain", "DrainParams"),
        "ael": ("logai.algorithms.parsing_algo.ael", "AEL", "AELParams"),
        "iplom": ("logai.algorithms.parsing_algo.iplom", "IPLoM", "IPLoMParams"),
    }

    def __init__(self):
        # directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        # zp = zipfile.ZipFile(os.path.join(directory, "cluster_patterns.json.zip"))
        # self.log_pattern_df = pd.read_json(zp.open("cluster_patterns.json"), orient="columns")
        # zf = zipfile.ZipFile(os.path.join(directory, "logline_map.csv.zip"))
        # self.logline_df = pd.read_csv(zf.open("logline_map.csv"))[["logline", "cluster_label", "lrt"]]
        self.parsing_app = None
        self.attributes = None

    def json_to_config(self, json_config):
        config = json.loads(json_config)
        workflow_config = WorkFlowConfig.from_dict(config)
        return workflow_config

    def yaml_to_config(self, yaml_config):
        config = yaml.safe_load(yaml_config)
        workflow_config = WorkFlowConfig.from_dict(config)
        return workflow_config

    def execute_auto_parsing(self, config: WorkFlowConfig):
        """
        This function executes the auto log parsing application.
        :param config: WorkFlowConfig
            Sample config:
            -------------------
            config = WorkFlowConfig(
                data_loader_config=DataLoaderConfig(
                    filepath=LOG_PATH,
                    log_type='csv',
                    dimensions=dict({
                        "attributes": ["Action", "ID"],
                        "body": ["Details"]
                    }),
                    reader_arg={
                        "header": None,
                        "sep": "|",
                        "on_bad_lines": "skip",
                        "names": ["Timestamps", "Action", "ID", "Details"]
                    }
                ),
                feature_extractor_config=FeatureExtractorConfig(),
                preprocessor_config=PreprocessorConfig(
                    custom_delimiters_regex=None
                ),
                log_parser_config=LogParserConfig(
                    parsing_algorithm='drain',
                    parsing_algo_params=DrainParams(
                        sim_th=0.4
                    )
                ),
            )
            -----------------------
        :return:
        """

        self.parsing_app = AutoLogSummarization(config)
        self.parsing_app.execute()

    def get_log_parsing_patterns(self):
        try:
            return self.parsing_app.parsing_results
        except Exception as e:
            logging.error("Cannot retrieve parsing result. Exception: {}".format(e))

    def get_log_patterns(self, attributes: dict):
        """
        For a given combination of dimension['attributes'], return the corresponding pattern
        :param attributes: list of attributes.
        :return: pd.Series of patterns
        """
        parsed_df = self.parsing_app.parsing_results
        if not attributes:
            res = parsed_df["parsed_logline"]
            return res

        for k, v in attributes.items():
            parsed_df = parsed_df[parsed_df[k] == v]

        res = parsed_df["parsed_logline"].unique()
        return list(res)

    def get_dynamic_parameter_list(self, log_pattern):
        """
        Give a log pattern, find the dynamic list of it in parsed_results
        :param log_pattern: str: log pattern
        :return: pd.Dataframe of dynamic parameter_list
        """
        try:
            dynamic_parameters = self.parsing_app.get_parameter_list(log_pattern)
        except Exception as e:
            raise RuntimeError(
                "Failed to get dynamic paramters for log_pattern: {} with exception {}".format(
                    log_pattern, e
                )
            )
        return dynamic_parameters

    def get_log_lines(self, log_pattern):
        """
        Give a log pattern, find all loglines with this pattern in parsed_results
        :param log_pattern: str: log pattern
        :return: pd.Series of loglines
        """
        df = self.result_table
        res = df[df["parsed_logline"] == log_pattern].drop(
            ["parameter_list", "parsed_logline"], axis=1
        )

        return res

    def get_attributes(self):
        return self.parsing_app.attributes

    @property
    def result_table(self):
        return self.parsing_app.parsing_results

    def summary_graph_df(self, attributes=[]):
        parsed_df = self.result_table

        if len(attributes) > 0:
            for attr in attributes:
                for k, v in attr.items():
                    parsed_df = parsed_df[parsed_df[k] == v]

        count_table = parsed_df["parsed_logline"].value_counts()

        scatter_df = pd.DataFrame(count_table)

        scatter_df.columns = ["counts"]

        scatter_df["ratio"] = scatter_df["counts"] * 1.0 / sum(scatter_df["counts"])
        scatter_df["order"] = np.array(range(scatter_df.shape[0]))

        return scatter_df
