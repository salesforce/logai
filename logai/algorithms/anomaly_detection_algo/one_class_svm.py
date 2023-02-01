#
# Copyright (c) 2023 Salesforce.com, inc.
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
from logai.algorithms.factory import factory


@dataclass
class OneClassSVMParams(Config):
    """Parameters for OneClass SVM based Anomaly Detector. For more explanations about the parameters see
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html.

    :param kernel: Specifies the kernel type to be used in the algorithm, i.e.,
        ``{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}``.
    :param degree: Degree of the polynomial kernel function (‘poly’).
    :param gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
    :param coef0: Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
    :param tol: Tolerance for stopping criterion.
    :param nu: An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
    :param shrinking: Whether to use the shrinking heuristic.
    :param cache_size: Specify the size of the kernel cache (in MB).
    :param verbose: Enable verbose output.
    """
    kernel: str = "linear"
    degree: int = 3
    gamma: str = "auto"
    coef0: float = 0.0
    tol: float = 1e-3
    nu: float = 0.5
    shrinking: bool = True
    cache_size: float = 200
    verbose: bool = False


@factory.register("detection", "one_class_svm", OneClassSVMParams)
class OneClassSVMDetector(AnomalyDetectionAlgo):
    def __init__(self, params: OneClassSVMParams):
        """
        OneClass SVM based Anomaly Detector. This is a wrapper class for the OneClassSVM model from scikit-learn library. For more details see 
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html.
        
        :param params: The parameters to control one class SVM models.
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

    def fit(self, log_features: pd.DataFrame):
        """
        Fit method to train the OneClassSVM on log data.

        :param log_features: Training log features as pandas DataFrame object.
        :return: The scores of the training dataset.
        """
        self.model.fit(log_features)
        train_scores = self.model.score_samples(log_features)
        train_scores = pd.DataFrame(train_scores, index=log_features.index)
        train_scores["trainval"] = True
        return train_scores

    def predict(self, log_features: pd.DataFrame) -> pd.Series:
        """
        Predict method to detect anomalies using OneClassSVM model on test log data.

        :param log_features: Test log features data as pandas DataFrame object.
        :return: A pandas dataframe of the predicted anomaly scores.
        """
        test_scores = self.model.predict(log_features)
        test_scores = pd.DataFrame(
            pd.Series(test_scores, index=log_features.index, name="anom_score")
        )
        test_scores["trainval"] = False
        return test_scores
