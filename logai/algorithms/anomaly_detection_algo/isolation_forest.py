#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd
from attr import dataclass

from sklearn.ensemble import IsolationForest

from logai.algorithms.algo_interfaces import AnomalyDetectionAlgo
from logai.config_interfaces import Config
from logai.algorithms.factory import factory


@dataclass
class IsolationForestParams(Config):
    n_estimators: int = 100
    max_samples: str = "auto"
    contamination: str = "auto"
    max_features: float = 1.0
    bootstrap: bool = False
    n_jobs: int = None
    random_state: object = None
    verbose: int = 0
    warm_start: bool = False


@factory.register("detection", "isolation_forest", IsolationForestParams)
class IsolationForestDetector(AnomalyDetectionAlgo):
    def __init__(self, params: IsolationForestParams):
        self.model = IsolationForest(
            n_estimators=params.n_estimators,
            max_samples=params.max_samples,
            contamination=params.contamination,
            max_features=params.max_features,
            bootstrap=params.bootstrap,
            n_jobs=params.n_jobs,
            random_state=params.random_state,
            verbose=params.verbose,
            warm_start=params.verbose,
        )

    def fit(self, log_features: pd.DataFrame):
        """
        Fit model
        :param log_features: pandas.DataFrame: input for model training
        :return: pandas.DataFrame
        """
        self.model.fit(log_features)
        train_scores = self.model.score_samples(log_features)
        train_scores = pd.DataFrame(train_scores, index=log_features.index)
        train_scores['trainval'] = True
        return train_scores

    def predict(self, log_features: pd.DataFrame) -> pd.Series:
        """
        Predict for input.
        :param log_features: pandas.DataFrame: input for inference
        :return: pandas.DataFrame
        """
        test_scores = self.model.predict(log_features)
        test_scores = pd.DataFrame(pd.Series(test_scores, index=log_features.index, name='anom_score'))

        test_scores['trainval'] = False
        return test_scores
