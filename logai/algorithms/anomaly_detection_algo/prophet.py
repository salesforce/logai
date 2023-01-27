#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd
from attr import dataclass

from logai.algorithms.algo_interfaces import AnomalyDetectionAlgo
from logai.config_interfaces import Config
from logai.algorithms.factory import factory


@dataclass
class ProphetParams(Config):
    """
    Prophet Anomaly Detector Params
    """

    pass


@factory.register("detection", "prophet", ProphetParams)
class ProphetDetector(AnomalyDetectionAlgo):
    """
    Prophet Anomaly Detector
    """

    def __init__(self):
        pass

    def fit(self, log_feature: pd.DataFrame):
        return

    def predict(self, log_feature: pd.DataFrame):
        return
