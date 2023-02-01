#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from logai.algorithms.algo_interfaces import CategoricalEncodingAlgo


class LabelEncoding(CategoricalEncodingAlgo):
    """This is a wrapper class for LabelEncoder from scikit-learn library. For more details see 
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html.
    """
    def __init__(self):
        """
        Initializes the label encoder.
        """
        self.model = LabelEncoder()

    def fit_transform(self, log_attributes: pd.DataFrame):
        """
        Fits and transforms log_attributes into label encoding categories.

        :param log_attributes: A list of log attributes in text format.
        :return: The label encoding categories.
        """

        res = pd.DataFrame()
        for feature_name in log_attributes.columns:
            x = self.model.fit_transform(log_attributes[feature_name])
            x_name = "{}_categorical".format(feature_name)
            res[x_name] = pd.Series(x, index=log_attributes[feature_name].index)
        return res
