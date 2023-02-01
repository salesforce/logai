#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
"""
Module that includes common data manipulation functions to be applied by pandas dataframes.
"""

import numpy as np
import pandas as pd
from merlion.utils import TimeSeries

from logai.utils import constants


def pad(x, max_len: np.array, padding_value: int = 0):
    """Method to trim or pad any 1-d numpy array to a given max length with the given padding value
    
    :param x: (np.array): given 1-d numpy array to be padded/trimmed
    :param max_len: (int): maximum length of padded/trimmed output
    :param padding_value: (int, optional): padding value. Defaults to 0.
    :return: np.array: padded/trimmed numpy array
    """
    flattened_vector = x
    fill_size = max_len - len(flattened_vector)
    if fill_size > 0:
        fill_zeros = np.full(fill_size, fill_value=padding_value)
        return np.concatenate((flattened_vector, fill_zeros), axis=0)
    else:
        return flattened_vector[:max_len]


def get_parameter_list(row):
    parameter_list = []
    if not isinstance(row[constants.LOGLINE_NAME], str) or not isinstance(
        row[constants.PARSED_LOGLINE_NAME], str
    ):
        return parameter_list
    ll = row[constants.LOGLINE_NAME].split()
    pp = row[constants.PARSED_LOGLINE_NAME].split()
    buffer = []

    i = 0
    j = 0
    consec_pattern = False
    while i < len(ll) and j < len(pp):
        # print(ll[i], pp[j])
        if ll[i] == pp[j]:
            if buffer:
                parameter_list.append(" ".join(buffer))
                buffer = []
            consec_pattern = False
            i += 1
            j += 1
        elif pp[j] == "*":
            if consec_pattern:
                parameter_list.append(" ".join(buffer))
                buffer = [ll[i]]
            else:
                buffer.append(ll[i])
            consec_pattern = True
            i += 1
            j += 1
        else:
            buffer.append(ll[i])
            i += 1
    if buffer:
        if i < len(ll):
            parameter_list.append(" ".join(buffer + ll[i:]))
        else:
            parameter_list.append(" ".join(buffer))
    return parameter_list


def pd_to_timeseries(log_features: pd.Series):
    """
    Convert pandas.DataFrame to merlion.TimeSeries for log counter vectors.

    :param log_features: log feature dataframe must only contain two columns
      ['timestamp': datetime, constants.LOGLINE_COUNTS: int].
    :return: merlion.TimeSeries type.
    """
    ts_df = log_features[constants.LOG_COUNTS]
    ts_df.index = log_features[constants.LOG_TIMESTAMPS]
    time_series = TimeSeries.from_pd(ts_df)
    return time_series
