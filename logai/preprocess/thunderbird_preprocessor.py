#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from logai.preprocess.openset_preprocessor import OpenSetPreprocessor
from logai.dataloader.data_model import LogRecordObject
from logai.preprocess.preprocessor import PreprocessorConfig
import pandas as pd
from logai.utils import constants


class ThunderbirdPreprocessor(OpenSetPreprocessor):
    """Custom Preprocessor for Open log dataset Thunderbird
    """

    def __init__(self, config: PreprocessorConfig):
        super().__init__(config)

    def _get_ids(self, logrecord: LogRecordObject) -> pd.Series:
        """get ids of loglines

        :param logrecord: (LogRecordObject): logrecord object
        :return: pd.Series: pandas series containing the ids of te loglines
        """
        return logrecord.span_id[constants.SPAN_ID]

    def _get_labels(self, logrecord: LogRecordObject):
        """get anomaly detection labels of loglines
        
        :param logrecord: (LogRecordObject):  logrecord object containing hdfs data
        :return:pd.Series: containing the anomaly detection labels of loglines
        """
        return logrecord.labels[constants.LABELS].apply(lambda x: int(x != "-"))
