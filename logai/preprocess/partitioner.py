#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd

from attr import dataclass

from logai.config_interfaces import Config
from logai.utils import constants


@dataclass
class PartitionerConfig(Config):
    """Config class for Partitioner. 

    :param group_by_category: The list of fields to group log data by .
    :param group_by_time: The string-type argument to specify grouping by time, supported types
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.
    :param sliding_window: The sliding window length if partitioning loglines into sliding windows.
    :param sep_token: The separator token string to be used as delimiter, when grouping log data .
    :param exclude_last_window: A boolean (default false) whether to exclude the last window
        when doing sliding window based grouping of log data.
    :param exclude_smaller_windows: A boolean (default false) whether to exclude windows of
        length smaller than the given `sliding_window` argument.
    """
    group_by_category: list = None
    group_by_time: str = None
    sliding_window: int = 0
    sep_token: str = "[SEP]"
    exclude_last_window: bool = False
    exclude_smaller_windows: bool = False


def concat_logs(windows, tokens):
    partitioned_loglines = []
    for window in windows:
        partitioned_loglines.append(tokens.join(window))

    return partitioned_loglines


class Partitioner:
    def __init__(self, config: PartitionerConfig):
        self.config = config
        return

    def sliding_window(self, loglines: pd.Series) -> pd.Series:
        """
        Conducts sliding window log partitioning.
        
        :param loglines: The series of loglines.
        :return: The series of logline sequence after sliding window.
        """
        partitioned_loglines = self._sliding_window(loglines)
        return pd.Series(partitioned_loglines, name=loglines.name)

    def group_counter(self, logrecord_df: pd.DataFrame) -> pd.DataFrame:
        """
        Groups log records by given categories and return counter vectors.
        
        :param logrecord_df: The log record dataframe.
        :return: The log counter vector dataframe after grouping.
        """
        if not self._valid_columns(logrecord_df.columns.values):
            raise ValueError("Make sure logrecord has the columns to group by.")

        group_by_timestamps = self.config.group_by_time
        group_by_category = self.config.group_by_category

        if group_by_category is not None:
            selected_df = logrecord_df[group_by_category]
            group_bys = group_by_category

        else:
            selected_df = logrecord_df
            group_bys = logrecord_df.columns.drop(constants.LOG_TIMESTAMPS)

        if group_by_timestamps is not None:
            selected_df[constants.LOG_TIMESTAMPS] = logrecord_df[
                constants.LOG_TIMESTAMPS
            ]
            group_bys += [
                pd.Grouper(
                    key=constants.LOG_TIMESTAMPS,
                    freq=group_by_timestamps,
                    offset=0,
                    label="left",
                )
            ]

        grouped_df = (
            logrecord_df.groupby(by=group_bys, as_index=False)
            .size()
            .rename(columns={"size": constants.LOG_COUNTS})
        )

        return grouped_df

    def group_sliding_window(
        self, logrecord_df: pd.DataFrame, logline_col_name=constants.LOGLINE_NAME
    ) -> pd.DataFrame:
        """
        Groups log records by sliding window based on the sliding window length, and returns
        the resulting pandas dataFrame object.
        
        :param logrecord_df: A pandas dataFrame on which grouping is to be applied.
        :return: A pandas dataFrame after sliding window based grouping.
        """
        if not self._valid_columns(logrecord_df.columns):
            raise ValueError("Make sure logrecord has the columns to group by.")

        if logline_col_name not in logrecord_df.columns:
            raise ValueError(
                "Logline column {} cannot be found in logrecord_df.columns".format(
                    logline_col_name
                )
            )

        sliding_window = self.config.sliding_window
        group_by_timestamps = self.config.group_by_time
        group_by_category = self.config.group_by_category
        sep = self.config.sep_token

        res_sequences = []
        res_attributes = []
        partition_cols = []
        grouper = []

        if group_by_timestamps and group_by_category:
            partition_cols = [constants.LOG_TIMESTAMPS] + group_by_category
            grouper = [
                pd.Grouper(
                    key="timestamp", freq=group_by_timestamps, base=0, label="left"
                )
            ] + group_by_category

        elif group_by_timestamps:
            partition_cols = [constants.LOG_TIMESTAMPS]
            grouper = [
                pd.Grouper(
                    key="timestamp", freq=group_by_timestamps, base=0, label="left"
                )
            ]

        elif group_by_category:
            partition_cols = group_by_category
            grouper = group_by_category

        if grouper:
            # if grouping
            for group_values, data in logrecord_df.groupby(grouper):
                loglines = data[logline_col_name]
                if sliding_window >= 0:
                    partitioned_seq = self._sliding_window(loglines)
                else:
                    partitioned_seq = sep.join(loglines)
                res_sequences += partitioned_seq

                res_attributes += [group_values for _ in range(len(partitioned_seq))]
            res_df = pd.DataFrame(res_attributes, columns=partition_cols)
            res_df[logline_col_name] = res_sequences
        else:
            # if not grouping, run sliding window directly
            if sliding_window >= 0:
                res_sequences = self.sliding_window(logrecord_df[logline_col_name])
                res_df = pd.DataFrame(res_sequences)
            else:
                res_df = sep.join(logrecord_df[logline_col_name])

        return res_df

    def _valid_columns(self, columns: list):
        if self.config.group_by_time is not None:
            if constants.LOG_TIMESTAMPS not in columns:
                return False

        if self.config.group_by_category is not None:
            for col in self.config.group_by_category:
                if col not in columns:
                    return False

        return True

    def _sliding_window(self, loglines: pd.Series) -> list:
        if self.config.sliding_window <= 0:
            return list(loglines)
        if self.config.exclude_smaller_windows:
            if len(loglines) <= self.config.sliding_window:
                if self.config.exclude_last_window:
                    loglines = [self.config.sep_token.join(list(loglines)[:-1])]
                    return loglines
                else:
                    loglines = [self.config.sep_token.join(loglines)]
                    return loglines
        closed = None
        if self.config.exclude_last_window:
            closed = "left"
        windows = loglines.rolling(
            window=self.config.sliding_window,
            min_periods=self.config.sliding_window,
            closed=closed,
        )
        if self.config.exclude_smaller_windows:
            windows = list(
                filter(lambda x: len(x) >= self.config.sliding_window, windows)
            )
        windows = list(map(lambda x: self.config.sep_token.join(x), windows))
        return windows
