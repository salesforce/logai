#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import numpy as np
import pandas as pd
from attr import dataclass

from logai.config_interfaces import Config
from logai.dataloader.data_model import LogRecordObject


@dataclass
class PreprocessorConfig(Config):
    """Config class for Preprocessor.

    :param custom_delimiters_regex: A dictionary of delimiter regex patterns in raw log data.
    :param custom_replace_list: A list of tuples of custom replace patterns in raw log data.
        Each Tuple should be of form ('regex-pattern-to-replace', 'replaced-pattern').
    """
    custom_delimiters_regex: dict = None
    custom_replace_list: list = None


class Preprocessor:
    """
    Preprocess class that contains common preprocess methods.
    """

    def __init__(self, config: PreprocessorConfig):
        self.config = config

    def clean_log(self, loglines: pd.Series) -> pd.Series:
        """Cleans the input log data.

        :param loglines: The raw loglines data to be cleaned .
        :return:pd.Series: The cleaned loglines data .
        """
        cleaned_log = loglines
        terms = pd.DataFrame()
        if self.config.custom_delimiters_regex:
            for reg in self.config.custom_delimiters_regex:
                try:
                    cleaned_log = cleaned_log.replace(
                        to_replace=reg, value=" ", regex=True
                    )
                except:
                    raise RuntimeError(
                        "Cannot replace custom regex delimiter {}".format(reg)
                    )

        if self.config.custom_replace_list:
            for pair in self.config.custom_replace_list:
                # TODO: refactor to tuple or map.
                try:
                    pattern = pair[0]
                    replacement = pair[1]
                    terms[replacement] = cleaned_log.str.findall(pat=pattern)
                    cleaned_log = cleaned_log.replace(
                        to_replace=pattern, value=replacement, regex=True
                    )
                except:
                    raise RuntimeError(
                        "Cannot replace custom regex: {} values: {}".format(
                            pair[0], pair[1]
                        )
                    )
        return cleaned_log, terms

    def group_log_index(self, attributes: pd.DataFrame, by: np.array) -> pd.DataFrame:
        """Groups log attributes (DataFrame) by a list of its fields.

        :param attributes: The log attribute data to be grouped.
        :param by: A list of fields of the log attribute DataFrame object to group by.
        :return: The log attribute data after grouping.
        """
        attributes["group_index"] = attributes.index
        group_index_list = (
            attributes.groupby(by=by).group_index.apply(np.array).reset_index()
        )

        return group_index_list

    def identify_timestamps(self, logrecord: LogRecordObject):
        pass
