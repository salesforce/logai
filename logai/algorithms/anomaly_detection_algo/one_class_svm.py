#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd
from attr import dataclass
from sklearn.svm import OneClassSVM

from logai.algorithms.algo_interfaces import AnomalyDetectionAlgo
from logai.config_interfaces import Config


@dataclass
class OneClassSVMParams(Config):
    kernel: str = "linear"
    degree: int = 3
    gamma: str = "auto"
    coef0: float = 0.0
    tol: float = 1e-3
    nu: float = 0.5
    shrinking: bool = True
    cache_size: float = 200
    verbose: bool = -1

    def from_dict(self, config_dict):
        super().from_dict(config_dict)

        return


class OneClassSVMDetector(AnomalyDetectionAlgo):
    def __init__(self, params: OneClassSVMParams):
        """

        :param params: OneClassSVMParams: parameters to control one class SVM models
        """
        self.model = OneClassSVM(
            kernel=params.kernel,
            degree=params.degree,
            gamma=params.gamma,
            coef0=params.coef0,
            tol=params.tol,
            nu=params.nu,
            shrinking=params.shrinking,
            cache_size=params.cache_size,
            verbose=params.verbose,
        )
        return

    def fit(self, log_features: pd.DataFrame):
        """

        :param log_features:
        :return:
        """
        self.model.fit(log_features)
        train_scores = self.model.score_samples(log_features)
        train_scores = pd.DataFrame(train_scores, index=log_features.index)
        train_scores['trainval'] = True
        return train_scores

    def predict(self, log_features: pd.DataFrame) -> pd.Series:
        """

        :param log_features:
        :return:
        """
        test_scores = self.model.predict(log_features)
        test_scores = pd.DataFrame(pd.Series(test_scores, index=log_features.index, name='anom_score'))
        test_scores['trainval'] = False
        return test_scores
