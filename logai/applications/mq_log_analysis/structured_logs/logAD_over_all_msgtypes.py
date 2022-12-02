#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from logai.analysis.anomaly_detector import AnomalyDetectionConfig, AnomalyDetector
from logai.dataloader.data_loader import FileDataLoader
from logai.dataloader.data_loader import DataLoaderConfig
from logai.information_extraction.feature_extractor import (
    FeatureExtractor,
    FeatureExtractorConfig,
)
from logai.algorithms.anomaly_detection_algo.local_outlier_factor import (
    LOFParams,
    LOFDetector,
)
from logai.algorithms.anomaly_detection_algo.isolation_forest import (
    IsolationForestParams,
)
from logai.information_extraction.categorical_encoder import (
    CategoricalEncoderConfig,
    CategoricalEncoder,
)
import pandas as pd
import numpy as np
import os

from collections import Counter

############### DATA LOADING START ###############

data_dir = "../datasets/mq/"
incident_dir = "na123_rac_7_2022-03-12"
log_record_type = "mqfrm"

data_dir = os.path.join(data_dir, log_record_type, incident_dir)

incident_filepath = os.path.join(data_dir, "incident_data.csv")
reference_filepath = os.path.join(data_dir, "reference_data.csv")


log_msgtype = "MessageTypeName"
log_metric = "WallClockTime"
log_datetime_field = "_time"
log_attribute_fields = [log_msgtype, log_metric]

dimensions = {
    "timestamp": [log_datetime_field],
    "attributes": log_attribute_fields,  # messagetypename, wallclocktime
    "body": [],
}


incident_file_config = DataLoaderConfig(
    filepath=incident_filepath,
    log_type="csv",
    dimensions=dimensions,
    # custom_delimeter_regex=custom_delimeter_regex,
    header=0,
)

incident_dataloader = FileDataLoader(incident_file_config)
incident_logrecord = incident_dataloader.load_data()

count_partitioning_config = FeatureExtractorConfig(
    group_by_time="1min", group_by_category=[log_msgtype]  # messagetype
)


reference_file_config = DataLoaderConfig(
    filepath=reference_filepath,
    log_type="csv",
    dimensions=dimensions,
    # custom_delimeter_regex=custom_delimeter_regex,
    header=0,
)

reference_dataloader = FileDataLoader(reference_file_config)
reference_logrecord = reference_dataloader.load_data()
print("Finished Data loading")
############### DATA LOADING END ###############


############### FREQ-COUNT BASED LOG ANOMALY DETECTION START ###############
print("Starting FREQ-COUNT BASED LOG ANOMALY DETECTION")
encoder_config = CategoricalEncoderConfig()
encoder = CategoricalEncoder(encoder_config)


count_feature_extractor = FeatureExtractor(count_partitioning_config)

incident_attributes = incident_logrecord.attributes
incident_timestamps = pd.to_datetime(incident_logrecord.timestamp["_time"])

reference_attributes = reference_logrecord.attributes
reference_timestamps = pd.to_datetime(reference_logrecord.timestamp["_time"])

incident_counts = count_feature_extractor.convert_to_counter_vector(
    log_pattern=None, attributes=incident_attributes, timestamps=incident_timestamps
)
reference_counts = count_feature_extractor.convert_to_counter_vector(
    log_pattern=None, attributes=reference_attributes, timestamps=reference_timestamps
)

incident_counts = incident_counts[[log_msgtype, constants.LOGLINE_COUNTS]]
reference_counts = reference_counts[[log_msgtype, constants.LOGLINE_COUNTS]]

incident_counts[log_msgtype] = encoder.fit_transform(incident_counts)[
    log_msgtype + "_categorical"
]
reference_counts[log_msgtype] = encoder.fit_transform(reference_counts)[
    log_msgtype + "_categorical"
]

lof_params = LOFParams()
lof_params.contamination = 0.001
lof_params.n_neighbors = 20
lof_params.novelty = True
ad_config = AnomalyDetectionConfig(algo_name="isolation_forest", algo_params=lof_params)
anomaly_detector = LOFDetector(lof_params)
anomaly_detector.fit(reference_counts)
anomaly_scores_counts = anomaly_detector.predict(incident_counts)
print("Anomaly Detection using LOF on Freq Count Stats")
print(anomaly_scores_counts)

isf_params = IsolationForestParams()
ad_config = AnomalyDetectionConfig(algo_name="isolation_forest", algo_params=isf_params)
anomaly_detector = AnomalyDetector(ad_config)

anomaly_detector.fit(reference_counts)
anomaly_scores_counts = anomaly_detector.predict(incident_counts)

print("Anomaly Detection using Isolation Forest on Freq Count Stats")
print(anomaly_scores_counts)

############### FREQ-COUNT BASED LOG ANOMALY DETECTION END ###############


############### METRIC VALUE BASED LOG ANOMALY DETECTION START ###############

print("Starting METRIC VALUE BASED LOG ANOMALY DETECTION")
metric_partitioning_config = FeatureExtractorConfig(
    group_by_category=log_attribute_fields  # messagetype
)

metric_feature_extractor = FeatureExtractor(metric_partitioning_config)

# Want to extract to create feature vector based on some structured attributes (which i dont want to group by), grouped by log record type/message type?
# When creating the feature vector, lib was not accepting empty log_vector ---> changed this in lib code
incident_metrics_count = metric_feature_extractor.convert_to_counter_vector(
    None, attributes=incident_attributes, timestamps=incident_timestamps
)
reference_metrics_count = metric_feature_extractor.convert_to_counter_vector(
    None, attributes=reference_attributes, timestamps=reference_timestamps
)

encoder_config = CategoricalEncoderConfig()
encoder = CategoricalEncoder(encoder_config)

incident_metrics_count[log_msgtype] = encoder.fit_transform(incident_metrics_count)[
    log_msgtype + "_categorical"
]
reference_metrics_count[log_msgtype] = encoder.fit_transform(reference_metrics_count)[
    log_msgtype + "_categorical"
]

lof_params = LOFParams()
lof_params.contamination = 0.001
lof_params.n_neighbors = 20
lof_params.novelty = True
ad_config = AnomalyDetectionConfig(algo_name="isolation_forest", algo_params=lof_params)
anomaly_detection = AnomalyDetector(ad_config)
anomaly_detector.fit(reference_metrics_count)
anomaly_scores = anomaly_detector.predict(incident_metrics_count)
print("Anomaly Detection using LOF on " + log_metric + " Stats")
print(anomaly_scores)


isf_params = IsolationForestParams()
ad_config = AnomalyDetectionConfig(algo_name="isolation_forest", algo_params=isf_params)
anomaly_detector = AnomalyDetector(ad_config)

anomaly_detector.fit(reference_metrics_count)
anomaly_scores = anomaly_detector.predict(incident_metrics_count)
print("Anomaly Detection using LOF on " + log_metric + " Stats")
print(anomaly_scores)

############### METRIC VALUE BASED LOG ANOMALY DETECTION END ###############
