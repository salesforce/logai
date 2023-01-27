from logai.preprocess.openset_preprocessor import OpenSetPreprocessor
from logai.dataloader.data_model import LogRecordObject
from logai.preprocess.preprocessor import PreprocessorConfig
import pandas as pd
import os
import yaml
from logai.utils import constants


class ThunderbirdPreprocessor(OpenSetPreprocessor):
    """Custom Preprocessor for Open log dataset Thunderbird

    Inherits:
        OpenSetPreprocessor: log preprocessor class for open log datasets
    """

    def __init__(self, config: PreprocessorConfig):
        super().__init__(config)

    def _get_ids(self, logrecord: LogRecordObject) -> pd.Series:
        """get ids of loglines

        Args:
            logrecord (LogRecordObject): logrecord object

        Returns:
            pd.Series: pandas series containing the ids of te loglines
        """
        return logrecord.span_id[constants.SPAN_ID]

    def _get_labels(self, logrecord: LogRecordObject):
        """get anomaly detection labels of loglines

        Args:
            logrecord (LogRecordObject):  logrecord object containing hdfs data

        Returns:
            pd.Series: containing the anomaly detection labels of loglines
        """
        return logrecord.labels[constants.LABELS].apply(lambda x: int(x != "-"))
