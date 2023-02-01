#
# Copyright (c) 2023 Salesforce.com, inc.
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
    """Parameters for isolation forest based anomaly detection. For more explanation of the parameters see the documentation page
    in https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html.

    :param n_estimators: The number of base estimators in the ensemble.
    :param max_samples: The number of samples to draw from X to train each base estimator.
    :param contamination: The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
    :param max_features: The number of features to draw from X to train each base estimator.
    :param bootstrap: If True, individual trees are fit on random subsets of the training data sampled with
        replacement. If False, sampling without replacement is performed.
    :param n_jobs: The number of jobs to run in parallel for both fit and predict.
    :param random_state: Controls the pseudo-randomness of the selection of the feature and split values
        for each branching step and each tree in the forest.
    :param verbose: Controls the verbosity of the tree building process.
    :param warm_start: When set to True, reuse the solution of the previous call to fit and add more estimators
        to the ensemble, otherwise, just fit a whole new forest.
    """
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
    """Isolation Forest based Anomaly Detector. This is a wrapper for the Isolation forest in scikit-learn library.
    """
    def __init__(self, params: IsolationForestParams):
        """Constructor for isolation forest based anomaly detector.
        
        :param params: An object of IsolationForestParams containing parameters of Isolation Forest.
        """
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
        Fits an isolation forest model.

        :param log_features: The input for model training.
        :return: The scores of the training dataset.
        """
        self.model.fit(log_features)
        train_scores = self.model.score_samples(log_features)
        train_scores = pd.DataFrame(train_scores, index=log_features.index)
        train_scores["trainval"] = True
        return train_scores

    def predict(self, log_features: pd.DataFrame) -> pd.Series:
        """
        Predicts anomalies.

        :param log_features: The input for inference.
        :return: A pandas dataframe of the predicted anomaly scores.
        """
        test_scores = self.model.predict(log_features)
        test_scores = pd.DataFrame(
            pd.Series(test_scores, index=log_features.index, name="anom_score")
        )

        test_scores["trainval"] = False
        return test_scores
