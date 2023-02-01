#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import torch
import os
import numpy as np
import pickle as pkl
import pandas as pd
from attr import dataclass

from logai.algorithms.algo_interfaces import VectorizationAlgo
from .semantic import Semantic, SemanticVectorizerParams
from .sequential import Sequential, SequentialVectorizerParams
from logai.config_interfaces import Config
from logai.dataloader.data_model import LogRecordObject
from logai.utils import constants
from logai.algorithms.factory import factory


class ForecastNNVectorizedDataset:
    """Class for storing vectorized dataset for forecasting based neural models.
    :param logline_features: (np.array): list of vectorized log-sequences
    :param labels: (list or pd.Series or np.array): list of labels (anomalous or non-anomalous) for each log sequence.
    :param nextlogline_ids: (list or pd.Series or np.array): list of ids of next loglines, for each log sequence
    :param span_ids: (list or pd.Series or np.array): list of ids of log sequences.
    """

    session_idx: str = "session_idx"
    features: str = "features"
    window_anomalies: str = "window_anomalies"
    window_labels: str = "window_labels"

    def __init__(self, logline_features, labels, nextlogline_ids, span_ids):
        
        self.dataset = []
        for data_i, label_i, next_i, index_i in zip(
            logline_features, labels, nextlogline_ids, span_ids
        ):
            self.dataset.append(
                {
                    self.session_idx: np.array([index_i]),
                    self.features: np.array(data_i),
                    self.window_anomalies: label_i,
                    self.window_labels: next_i,
                }
            )


@dataclass
class ForecastNNVectorizerParams(Config):
    """Config class for vectorizer for forecast based neural models for log representation learning.

    :param feature_type: The type of log feature representation where the supported types "semantics" and "sequential".
    :param label_type: The type of label, anomaly or next_log, which corresponds to the supervised and the
        forecasting based unsupervised setting.
    :param sep_token: The separator token used when constructing the log sequences during log grouping/partitioning. (default = "[SEP]")
    :param max_token_len: The maximum token length of the input.
    :param min_token_count: The minimum number of occurrences of a token in the training data, for it to be
        considered in the vocab.
    :param embedding_dim: The embedding dimension of the tokens.
    :param output_dir: The path to output directory where the vectorizer model directory and metadata file
        would be created.
    :param vectorizer_metadata_filepath: The path to file where the vectorizer metadata would be saved.
        This would be read by the anomaly detection model and should be set in the metadata_filepath of
        the forecast_nn based anomaly detector.
    :param vectorizer_model_dirpath: The path to directory containing the vectorizer model.
    """

    feature_type: str = None  # supported types "semantics" and "sequential"
    label_type: str = None
    sep_token: str = "[SEP]"
    max_token_len: int = None
    min_token_count: int = None
    embedding_dim: int = None
    output_dir: str = ""
    vectorizer_metadata_filepath: str = ""
    vectorizer_model_dirpath: str = ""

    sequentialvec_config: object = None
    semanticvec_config: object = None


@factory.register("vectorization", "forecast_nn", ForecastNNVectorizerParams)
class ForecastNN(VectorizationAlgo):
    """Vectorizer Class for forecast based neural models for log representation learning.
    
    :param config: config object specifying parameters of forecast based neural log repersentation learning model.
    """

    def __init__(self, config: ForecastNNVectorizerParams):
        self.meta_data = {}
        self.config = config
        self.config.vectorizer_model_dirpath = os.path.join(
            self.config.output_dir, "embedding_model"
        )
        self.config.vectorizer_metadata_filepath = os.path.join(
            self.config.vectorizer_model_dirpath, "metadata.pkl"
        )

        if not os.path.exists(self.config.vectorizer_model_dirpath):
            os.makedirs(self.config.vectorizer_model_dirpath)

        sequentialvec_config = SequentialVectorizerParams.from_dict(
            {
                "sep_token": self.config.sep_token,
                "max_token_len": self.config.max_token_len,
                "model_save_dir": self.config.vectorizer_model_dirpath,
            }
        )
        self.sequential_vectorizer = Sequential(sequentialvec_config)
        self.semantic_vectorizer = None
        if self.config.feature_type == "semantics":
            semanticvec_config_dict = {
                "max_token_len": self.config.max_token_len,
                "min_token_count": self.config.min_token_count,
                "sep_token": self.config.sep_token,
                "embedding_dim": self.config.embedding_dim,
                "model_save_dir": self.config.vectorizer_model_dirpath,
            }
            semanticvec_config = SemanticVectorizerParams.from_dict(
                semanticvec_config_dict
            )
            self.semantic_vectorizer = Semantic(semanticvec_config)

    def _process_logsequence(self, data_sequence):
        data_sequence = data_sequence.dropna()
        unique_data_instances = pd.Series(
            list(
                set(
                    self.config.sep_token.join(list(data_sequence)).split(
                        self.config.sep_token
                    )
                )
            )
        )
        return unique_data_instances

    def fit(self, logrecord: LogRecordObject):
        """Fit method to train vectorizer.

        :param logrecord: A log record object to train the vectorizer on.
        """
        if self.sequential_vectorizer.vocab is None or (
            self.config.feature_type == "semantics"
            and self.semantic_vectorizer.vocab is None
        ):
            loglines = logrecord.body[
                constants.LOGLINE_NAME
            ]  # data[self.config.loglines_field]
            nextloglines = logrecord.attributes[
                constants.NEXT_LOGLINE_NAME
            ]  # data[self.config.nextlog_field]
            loglines = pd.concat([loglines, nextloglines])
            loglines = self._process_logsequence(loglines)
        if self.sequential_vectorizer.vocab is None:
            self.sequential_vectorizer.fit(loglines)
        if (
            self.config.feature_type == "semantics"
            and self.semantic_vectorizer.vocab is None
        ):
            self.semantic_vectorizer.fit(loglines)
        self._dump_meta_data()

    def transform(self, logrecord: LogRecordObject):
        """Transform method to run vectorizer on logrecord object.

        :param logrecord: A log record object to be vectorized.
        :return: ForecastNNVectorizedDataset object containing the vectorized dataset.
        """
        if self.config.feature_type == "sequential":
            logline_features = self.sequential_vectorizer.transform(
                logrecord.body[constants.LOGLINE_NAME]
            )
        elif self.config.feature_type == "semantics":
            logline_features = self.semantic_vectorizer.transform(
                logrecord.body[constants.LOGLINE_NAME]
            )
        if constants.NEXT_LOGLINE_NAME in logrecord.attributes:
            nextlogline_ids = self.sequential_vectorizer.transform(
                logrecord.attributes[constants.NEXT_LOGLINE_NAME]
            ).apply(lambda x: x[0])
        else:
            nextlogline_ids = None
        labels = logrecord.labels[constants.LABELS]
        span_ids = logrecord.span_id[constants.SPAN_ID]
        samples = ForecastNNVectorizedDataset(
            logline_features=logline_features,
            labels=labels,
            nextlogline_ids=nextlogline_ids,
            span_ids=span_ids,
        )
        return samples

    def _dump_meta_data(self):
        if not os.path.exists(self.config.vectorizer_metadata_filepath):
            if self.config.feature_type == "sequential":
                self.meta_data["vocab_size"] = self.sequential_vectorizer.vocab_size
            else:
                self.meta_data["vocab_size"] = self.semantic_vectorizer.vocab_size
            if self.config.feature_type == "semantics":
                self.meta_data["pretrain_matrix"] = torch.Tensor(
                    self.semantic_vectorizer.embed_matrix
                )
            if self.config.label_type == "anomaly":
                self.meta_data["num_labels"] = 2
            else:
                self.meta_data["num_labels"] = self.sequential_vectorizer.vocab_size
            pkl.dump(
                self.meta_data, open(self.config.vectorizer_metadata_filepath, "wb")
            )
        else:
            self.meta_data = pkl.load(
                open(self.config.vectorizer_metadata_filepath, "rb")
            )
