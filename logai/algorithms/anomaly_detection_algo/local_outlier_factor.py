#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd
from attr import dataclass
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

from logai.algorithms.algo_interfaces import AnomalyDetectionAlgo
from logai.config_interfaces import Config
from logai.algorithms.factory import factory


@dataclass
class LOFParams(Config):
    """Parameters of Locality Outlier Factors based Anomaly Detector 
    For more explanations of the parameters see https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
    """
    n_neighbors: int = 20
    algorithm: str = "auto"
    leaf_size: int = 30
    metric: callable or str = "minkowski"
    p: int = 2
    metric_params: dict = None
    contamination: str = "auto"
    novelty: bool = True
    n_jobs: int = None


@factory.register("detection", "lof", LOFParams)
class LOFDetector(AnomalyDetectionAlgo):
    """Locality Outlier Factor based Anomaly Detector. This is a wrapper method for the LOF based Detector in scikit-learn library 
    See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html for more details
    """
    def __init__(self, params: LOFParams):
        self.model = LocalOutlierFactor(
            n_neighbors=params.n_neighbors,
            algorithm=params.algorithm,
            leaf_size=params.leaf_size,
            metric=params.metric,
            p=params.p,
            metric_params=params.metric_params,
            contamination=params.contamination,
            novelty=params.novelty,
            n_jobs=params.n_jobs,
        )

    def fit(self, log_features: pd.DataFrame):
        """
        Fit model
        
        :param log_features: pandas.DataFrame: input for model training
        :return: pandas.Dataframe
        """
        self.model.fit(
            np.array(log_features)
        )  # casting to numpy array to avoid warning on feature names
        train_scores = self.model.score_samples(log_features)
        train_scores = pd.DataFrame(train_scores, index=log_features.index)
        train_scores["trainval"] = True
        return train_scores

    def predict(self, log_features: pd.DataFrame) -> pd.Series:
        """
        Predict for input
        
        :param log_features: pandas.DataFrame: input for inference
        :return: pandas.Dataframe
        """
        test_scores = self.model.predict(
            np.array(log_features)
        )  # casting to numpy array to avoid warning on feature names
        test_scores = pd.DataFrame(
            pd.Series(test_scores, index=log_features.index, name="anom_score")
        )

        test_scores["trainval"] = False
        return test_scores
