#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import numpy as np
import pandas as pd
from attr import dataclass

from logai.algorithms.algo_interfaces import AnomalyDetectionAlgo
from logai.config_interfaces import Config
from logai.algorithms.factory import factory


def compute_probs(data, n=10):
    h, e = np.histogram(data, n)
    p = h / np.linalg.norm(h, ord=1)
    return e, p


def support_intersection(p, q):
    sup_int = list(filter(lambda x: (x[0] != 0) & (x[1] != 0), zip(p, q)))
    return sup_int


def get_probs(list_of_tuples):
    p = np.array([p[0] for p in list_of_tuples])
    q = np.array([p[1] for p in list_of_tuples])
    return p, q


def kl_divergence(p, q):
    p[p == 0] = 1e-20
    q[q == 0] = 1e-20
    return np.sum(p * np.log(p / q))


def js_divergence(p, q):
    m = (1.0 / 2.0) * (p + q)
    return (1.0 / 2.0) * kl_divergence(p, m) + (1.0 / 2.0) * kl_divergence(q, m)


@dataclass
class DistributionDivergenceParams(Config):
    n_bins: int = 100
    type: list = ["KL"]  # "KL", "JS", "KL,JS"


@factory.register("detection", "distribution_divergence", DistributionDivergenceParams)
class DistributionDivergence(AnomalyDetectionAlgo):
    def __init__(self, params: DistributionDivergenceParams):
        self.n_bins = params.n_bins
        self.type = params.type
        if any([x not in ["KL", "JS", "KL,JS"] for x in self.type]):
            raise Exception("Type of distribution divergence allowed are: KL, JS")
        self.train_sample = None

    def fit(self, log_features: pd.DataFrame):
        """
        K-L divergence model fitting. Learn the sample distribution of training data
        :return:
        """
        self.train_sample = np.array(log_features)

    def predict(self, log_features: pd.DataFrame) -> pd.Series:
        """
        Compute K-L divergence between training and testing samples. Return anomalous scores.
        :param log_features:
        :return:
        """

        log_features = np.array(log_features)
        if type(self.n_bins) == int:
            _, n_bins = np.histogram(
                np.vstack([np.array(self.train_sample), log_features]), self.n_bins
            )
        else:
            n_bins = self.n_bins
        e, p = compute_probs(self.train_sample, n=n_bins)
        _, q = compute_probs(log_features, n=e)

        # list_of_tuples = support_intersection(p, q)
        # p, q = get_probs(list_of_tuples)

        dist_divergence_scores = None
        if "KL" in self.type:
            kl = kl_divergence(p, q)
            dist_divergence_scores = kl
        if "JS" in self.type:
            js = js_divergence(p, q)
            if dist_divergence_scores is not None:
                dist_divergence_scores = (dist_divergence_scores, js)
            else:
                dist_divergence_scores = js
        res = pd.DataFrame(pd.Series(dist_divergence_scores).rename("anom_score"))
        res["trainval"] = False
        return dist_divergence_scores
