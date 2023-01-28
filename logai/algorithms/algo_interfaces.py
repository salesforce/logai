#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import abc

import pandas as pd
from logai.dataloader.data_model import LogRecordObject


class ParsingAlgo(abc.ABC):
    """
    Interface for parsing algorithms.

    """

    @abc.abstractmethod
    def fit(self, loglines: pd.Series):
        """
        fit parsing algorithm with input

        :param loglines: pd.Series of loglines as input
        :return: pd.Dataframe

        """
        pass

    @abc.abstractmethod
    def parse(self, loglines: pd.Series) -> pd.DataFrame:
        """
        parse loglines
        :param loglines: pd.Series of loglines to parse
        :return: pd.Dataframe of parsed results ["loglines", "parsed_loglines", "parameter_list"].
        """
        pass


class VectorizationAlgo(abc.ABC):
    """
    Interface for logline vectorization algorithms
    """

    @abc.abstractmethod
    def fit(self, loglines: pd.Series):
        """
        fit vectorizor with input.
        :param loglines:
        :return:
        """
        pass

    @abc.abstractmethod
    def transform(self, loglines: pd.Series) -> pd.DataFrame:
        """
        transform given loglines into vectors.
        :param loglines:
        :return:
        """
        pass


class FeatureExtractionAlgo(abc.ABC):
    """
    Interface for feature extraction algorithms
    """

    @abc.abstractmethod
    def extract(self):
        pass


class ClusteringAlgo(abc.ABC):
    """
    Interface for clustering algorithms
    """

    @abc.abstractmethod
    def fit(self, log_features: pd.DataFrame):
        pass

    @abc.abstractmethod
    def predict(self, log_features: pd.DataFrame):
        pass


class AnomalyDetectionAlgo(abc.ABC):
    """
    Interface for clustering algorithms
    """

    @abc.abstractmethod
    def fit(self, log_features: pd.DataFrame):
        pass

    @abc.abstractmethod
    def predict(self, log_features: pd.DataFrame) -> pd.Series:
        pass


class NNAnomalyDetectionAlgo(abc.ABC):
    @abc.abstractmethod
    def fit(self, train_data, dev_data: LogRecordObject):
        pass

    @abc.abstractmethod
    def predict(self, test_data: LogRecordObject):
        pass


class CategoricalEncodingAlgo(abc.ABC):
    """
    Interface for categorical encoders
    """

    @abc.abstractmethod
    def fit_transform(self, log_attributes: pd.DataFrame) -> pd.DataFrame:
        pass
