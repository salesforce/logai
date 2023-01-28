#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import json
import yaml

from logai.applications.application_interfaces import WorkFlowConfig
from logai.applications.log_anomaly_detection import LogAnomalyDetection
from .utils import ParamInfoMixin


class LogAnomaly(ParamInfoMixin):
    algorithms = {
        "one_class_svm": (
            "logai.algorithms.anomaly_detection_algo.one_class_svm",
            "OneClassSVMDetector",
            "OneClassSVMParams",
        ),
        "isolation_forest": (
            "logai.algorithms.anomaly_detection_algo.isolation_forest",
            "IsolationForestDetector",
            "IsolationForestParams",
        ),
        "lof": (
            "logai.algorithms.anomaly_detection_algo.local_outlier_factor",
            "LOFDetector",
            "LOFParams",
        ),
        "distribution_divergence": (
            "logai.algorithms.anomaly_detection_algo.distribution_divergence",
            "DistributionDivergence",
            "DistributionDivergenceParams",
        ),
        "dbl": (
            "logai.algorithms.anomaly_detection_algo.dbl",
            "DBLDetector",
            "DBLDetectorParams",
        ),
        "ets": (
            "logai.algorithms.anomaly_detection_algo.ets",
            "ETSDetector",
            "ETSDetectorParams",
        ),
    }

    def __init__(self):
        self.app = None
        self.attributes = None

    @property
    def results(self):
        return self.app.results

    def execute_anomaly_detection(self, config: WorkFlowConfig):
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

        self.app = LogAnomalyDetection(config)
        self.app.execute()

        return

    def get_anomalies(self, attributes=[]):

        df = self.get_results(attributes)
        df = df[df["is_anomaly"]]
        return df

    def get_results(self, attributes=[]):
        df = self.app.results
        if not attributes:
            return df

        for k, v in attributes.items():
            df = df.loc[df[k] == v]

        return df

    def get_attributes(self):
        return self.app.attributes

    def get_event_group(self):
        return self.app.event_group

    # TODO: merge with PD and move to utils
    def json_to_config(self, json_config):
        config = json.loads(json_config)
        workflow_config = WorkFlowConfig.from_dict(config)
        return workflow_config

    # TODO: merge with PD and move to utils
    def yaml_to_config(self, yaml_config):
        config = yaml.safe_load(yaml_config)
        workflow_config = WorkFlowConfig.from_dict(config)
        return workflow_config
