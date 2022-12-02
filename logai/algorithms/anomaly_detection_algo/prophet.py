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


@dataclass
class ProphetParams(Config):
    """
    Prophet Anomaly Detector Params
    """
    def from_dict(self, config_dict):
        super.from_dict(config_dict)

        return


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

