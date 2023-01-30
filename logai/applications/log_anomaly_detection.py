#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from logai.analysis.anomaly_detector import AnomalyDetector, AnomalyDetectionConfig
from logai.applications.application_interfaces import WorkFlowConfig
from logai.dataloader.data_loader import FileDataLoader
from logai.dataloader.data_model import LogRecordObject
from logai.dataloader.openset_data_loader import OpenSetDataLoader
from logai.information_extraction.categorical_encoder import (
    CategoricalEncoder,
    CategoricalEncoderConfig,
)
from logai.information_extraction.feature_extractor import FeatureExtractor
from logai.information_extraction.log_parser import LogParser
from logai.information_extraction.log_vectorizer import LogVectorizer
from logai.preprocess.partitioner import PartitionerConfig, Partitioner
from logai.preprocess.preprocessor import Preprocessor
from logai.utils import constants, evaluate


class LogAnomalyDetection:
    """This is a workflow for log anomaly detection. 
    """
    def __init__(self, config: WorkFlowConfig):
        self.config = config
        self._timestamps = pd.DataFrame()
        self._attributes = pd.DataFrame()
        self._feature_df = pd.DataFrame()
        self._counter_df = pd.DataFrame()
        self._loglines = pd.DataFrame()
        self._log_templates = pd.DataFrame()
        self._ad_results = pd.DataFrame()
        self._labels = pd.DataFrame()
        self._index_group = pd.DataFrame()
        self._loglines_with_anomalies = pd.DataFrame()
        self._group_anomalies = None

        return

    @property
    def timestamps(self):
        return self._timestamps

    @property
    def loglines(self):
        return self._loglines

    @property
    def log_templates(self):
        return self._log_templates

    @property
    def attributes(self):
        return self._attributes

    @property
    def results(self):
        res = (
            self._loglines_with_anomalies.join(self.attributes)
            .join(self.timestamps)
            .join(self.event_group)
        )
        return res

    @property
    def anomaly_results(self):

        return self.results[self.results["is_anomaly"]]

    @property
    def anomaly_labels(self):
        return self._labels

    @anomaly_labels.setter
    def anomaly_labels(self, labels):
        self._labels = labels

    @property
    def event_group(self):
        event_index_map = dict()
        for group_id, indices in self._index_group["event_index"].items():
            for i in indices:
                event_index_map[i] = group_id

        event_index = pd.Series(event_index_map).rename("group_id")
        return event_index

    @property
    def feature_df(self):
        return self._feature_df

    @property
    def counter_df(self):
        return self._counter_df

    def evaluation(self):
        if self.anomaly_labels is None:
            raise TypeError

        labels = self.anomaly_labels.to_numpy()
        pred = np.array([1 if r else 0 for r in self.results["is_anomaly"]])
        return evaluate.get_accuracy_precision_recall(labels, pred)

    def execute(self):
        logrecord = self._load_data()
        # Preprocessor cleans the loglines

        preprocessed_logrecord = self._preprocess(logrecord)

        # Parsing
        loglines = preprocessed_logrecord.body[constants.LOGLINE_NAME]
        parsed_loglines = self._parse(loglines)

        # Feature extraction
        feature_extractor = FeatureExtractor(self.config.feature_extractor_config)

        # Get Counter Set
        self._counter_df = feature_extractor.convert_to_counter_vector(
            timestamps=logrecord.timestamp[constants.LOG_TIMESTAMPS],
            attributes=self.attributes,
        )

        if self.config.anomaly_detection_config.algo_name in constants.COUNTER_AD_ALGO:
            self._counter_df["attribute"] = self._counter_df.drop(
                [constants.LOG_COUNTS, constants.LOG_TIMESTAMPS, constants.EVENT_INDEX],
                axis=1,
            ).apply(lambda x: "-".join(x.astype(str)), axis=1)

            attr_list = self._counter_df["attribute"].unique()
            res = pd.Series()
            for attr in attr_list:
                temp_df = self._counter_df[self._counter_df["attribute"] == attr]
                if temp_df.shape[0] < constants.MIN_TS_LENGTH:
                    anom_score = np.repeat(0.0, temp_df.shape[0])
                    res = res.append(pd.Series(anom_score, index=temp_df.index))
                else:
                    train, test = train_test_split(
                        temp_df[[constants.LOG_TIMESTAMPS, constants.LOG_COUNTS]],
                        shuffle=False,
                        train_size=0.7,
                    )
                    anomaly_detector = AnomalyDetector(
                        self.config.anomaly_detection_config
                    )
                    anom_score_training = pd.Series(
                        np.repeat(0.0, train.shape[0]), index=train.index
                    )
                    anomaly_detector.fit(train)
                    anom_score = anomaly_detector.predict(test)

                    res = res.append(anom_score_training)
                    res = res.append(anom_score["anom_score"])
            self._ad_results = pd.DataFrame(res.rename("result"))
            self._index_group = self._counter_df[[constants.EVENT_INDEX]]

        else:
            # Vectorization
            vectorizor = LogVectorizer(self.config.log_vectorizer_config)
            vectorizor.fit(parsed_loglines)

            # Log vector is a pandas.Series
            log_vectors_w2v = vectorizor.transform(parsed_loglines)

            # Categorical Encoding
            encoder = CategoricalEncoder(self.config.categorical_encoder_config)

            attributes = encoder.fit_transform(logrecord.attributes)
            attributes.columns = logrecord.attributes.columns

            # Get Feature Set
            index_group, feature_df = feature_extractor.convert_to_feature_vector(
                timestamps=logrecord.timestamp[constants.LOG_TIMESTAMPS],
                log_vectors=log_vectors_w2v,
                attributes=attributes,
            )

            self._index_group = index_group[["event_index"]]
            feature_for_anomaly_detection = feature_df.loc[
                :, ~feature_df.columns.isin([constants.LOG_TIMESTAMPS])
            ]
            anomaly_detector = AnomalyDetector(self.config.anomaly_detection_config)

            anomaly_detector.fit(feature_for_anomaly_detection)
            anomalies = anomaly_detector.predict(feature_for_anomaly_detection)[
                "anom_score"
            ]
            self._ad_results = pd.DataFrame(anomalies.rename("result"))

        anomaly_group_indices = self._ad_results[
            self._ad_results["result"] > 0.0
        ].index.values

        anomaly_indices = []

        for indices in self._index_group["event_index"].iloc[anomaly_group_indices]:
            anomaly_indices += indices

        df = pd.DataFrame(self.loglines)
        df["_id"] = df.index.values

        df["is_anomaly"] = [True if i in anomaly_indices else False for i in df["_id"]]
        self._loglines_with_anomalies = df

        return

    def _load_data(self):
        if self.config.open_set_data_loader_config is not None:
            dataloader = OpenSetDataLoader(self.config.open_set_data_loader_config)
            logrecord = dataloader.load_data()
        elif self.config.data_loader_config is not None:
            dataloader = FileDataLoader(self.config.data_loader_config)
            logrecord = dataloader.load_data()
        else:
            raise ValueError(
                "data_loader_config or open_set_data_loader_config is needed to load data."
            )
        return logrecord

    def _preprocess(self, log_record):
        logline = log_record.body[constants.LOGLINE_NAME]

        self._loglines = logline
        self._timestamps = log_record.timestamp
        self._attributes = log_record.attributes.astype(str)

        preprocessor = Preprocessor(self.config.preprocessor_config)
        preprocessed_loglines, _ = preprocessor.clean_log(logline)

        new_log_record = LogRecordObject(
            body=pd.DataFrame(preprocessed_loglines, columns=[constants.LOGLINE_NAME]),
            timestamp=log_record.timestamp,
            attributes=log_record.attributes,
        )

        return new_log_record

    def _parse(self, loglines):

        parser = LogParser(self.config.log_parser_config)
        parsed_results = parser.parse(loglines.dropna())

        parsed_loglines = parsed_results[constants.PARSED_LOGLINE_NAME]

        return parsed_loglines
