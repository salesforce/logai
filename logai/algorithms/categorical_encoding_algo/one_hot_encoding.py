#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from logai.algorithms.algo_interfaces import CategoricalEncodingAlgo
from logai.config_interfaces import Config


class OneHotEncodingParams(Config):
    """
    Configuration for One-Hot Encoding. For more details on the parameters see
     https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    """

    categories: str = "auto"
    drop: object = None
    dtype: np.float64 = np.float64
    handle_unknown: str = "error"


class OneHotEncoding(CategoricalEncodingAlgo):
    """This is a wrapper class for OneHotEncoder from scikit-learn library.
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    """
    def __init__(self, params: OneHotEncodingParams):
        """
        Init one-hot encoder
        """
        self.model = OneHotEncoder(
            categories=params.categories,
            drop=params.drop,
            sparse=False,  # Default return non-sparse matrix, need to think about how to handle sparse matrix
            dtype=params.dtype,
            handle_unknown=params.handle_unknown,
        )

    def fit_transform(self, log_attributes: pd.DataFrame) -> pd.DataFrame:
        """
        fit and transform log attributes into one-hot encoding categories
        
        :param log_attributes: list of log attributes in text form
        :return: pd.DataFrame: categories in on-hot encoding
        """
        col_names = log_attributes.columns
        if len(col_names) == 1:
            res_col_name_prefix = col_names[0]
        else:
            res_col_name_prefix = "-".join(col_names)
        self.model.fit(log_attributes)
        res = pd.DataFrame(
            self.model.transform(log_attributes), index=log_attributes.index
        )
        res_col_names = ["{}-{}".format(res_col_name_prefix, c) for c in res.columns]
        res.columns = res_col_names
        return res
