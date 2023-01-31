#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from logai.preprocess.openset_preprocessor import OpenSetPreprocessor
from logai.preprocess.preprocessor import PreprocessorConfig
import pandas as pd
from logai.utils import constants
from logai.dataloader.data_model import LogRecordObject


class BGLPreprocessor(OpenSetPreprocessor):
    """
    Custom preprocessor for Open log dataset BGL.

    Inherits:
        OpenSetPreprocessor: log preprocessor class for open log datasets.
    """

    def __init__(self, config: PreprocessorConfig):
        super().__init__(config)

    def _get_ids(self, logrecord: LogRecordObject) -> pd.Series:
        """get ids of loglines

        Args:
            logrecord (LogRecordObject):  logrecord object containing the BGL data

        Returns:
            pd.Series: containing the ids of the loglines
        """
        time_unit_in_secs = 60  # 21600.0 # 6 hours
        ids = logrecord.span_id[constants.SPAN_ID].astype(int)
        start_time = ids[0]
        session_ids = ids.apply(lambda x: int((x - start_time) / time_unit_in_secs))
        return session_ids

    def _get_labels(self, logrecord: LogRecordObject):
        """get anomaly detection labels of loglines

        Args:
            logrecord (LogRecordObject): logrecord object containing the BGL data

        Returns:
            pd.Series: containing the labels of the loglines
        """
        return logrecord.labels[constants.LABELS].apply(lambda x: int(x != "-"))
