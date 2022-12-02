#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from logai.dataloader.data_loader import FileDataLoader
from logai.dataloader.data_loader import DataLoaderConfig
from logai.information_extraction.feature_extractor import (
    FeatureExtractor,
    FeatureExtractorConfig,
)
from logai.algorithms.anomaly_detection_algo.local_outlier_factor import LOFParams
from logai.algorithms.anomaly_detection_algo.distribution_divergence import (
    DistributionDivergenceParams,
)
from logai.analysis.anomaly_detector import AnomalyDetectionConfig, AnomalyDetector
import pandas as pd
import numpy as np
import os

from collections import Counter

############### DATA LOADING START ###############

data_dir =  '../../datasets/mq/'
incident_dir = 'na123_rac_7_2022-03-12'
log_record_type = 'mqfrm'


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
    header=0
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


print("Starting FREQ-COUNT VALUE BASED LOG ANOMALY DETECTION")

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

incident_counts_msgtype = {x[0]:pd.DataFrame(x[1]['Counts']) for x in incident_counts.groupby(log_msgtype)}
reference_counts_msgtype = {x[0]:pd.DataFrame(x[1]['Counts']) for x in reference_counts.groupby(log_msgtype)}

num_incident_timechunks = max([len(x) for x in incident_counts_msgtype.values()])
num_reference_timechunks = max([len(x) for x in reference_counts_msgtype.values()])

incident_counts_msgtype = {k:v+[0]*(num_incident_timechunks - len(v)) for k,v in incident_counts_msgtype.items()}
reference_counts_msgtype = {k:v+[0]*(num_reference_timechunks - len(v)) for k,v in reference_counts_msgtype.items()}


lof_params = LOFParams()
lof_params.contamination = 0.001
lof_params.n_neighbors = 20
lof_params.novelty = True
all_msgtypes = set(incident_counts_msgtype).union(set(reference_counts_msgtype))
ad_config = AnomalyDetectionConfig(algo_name="lof", algo_params=lof_params)
anomaly_detector = AnomalyDetector(ad_config)


num_bins_freq_count = 100

all_counts = []
all_counts.extend(list(incident_counts[constants.LOGLINE_COUNTS]))
all_counts.extend(list(reference_counts[constants.LOGLINE_COUNTS]))
max_freq_counts_value = max(all_counts)
range_of_count_values = range(0, max_freq_counts_value + 1)
_, freq_count_bins = np.histogram(
    np.array(range_of_count_values), bins=num_bins_freq_count
)

distdiv_params = DistributionDivergenceParams(n_bins=freq_count_bins, type=["KL", "JS"])
ad_config = AnomalyDetectionConfig(
    algo_name="distribution_divergence", algo_params=distdiv_params
)
anomaly_ranker = AnomalyDetector(ad_config)

stats_per_msgtype = {}
for msgtype in all_msgtypes:
    if len(reference_counts_msgtype[msgtype]) < 2:
        continue

    if msgtype not in incident_counts_msgtype:
        incident_counts_msgtype[msgtype] = [0]*num_incident_timechunks

    anomaly_detector.fit(reference_counts_msgtype[msgtype])
    lof_prediction = anomaly_detector.predict(incident_counts_msgtype[msgtype])
    
    anomaly_ranker.fit(reference_counts_msgtype[msgtype])
    kl_div, js_div = anomaly_ranker.predict(incident_counts_msgtype[msgtype])


    stats = {'num_anomalies_freq_count': len(lof_prediction[lof_prediction==-1]), 'kl_div_freq_count': kl_div, 'js_div_freq_count': js_div}
    stats_per_msgtype[msgtype] = stats 


############### FREQ-COUNT BASED LOG ANOMALY DETECTION END ###############

############### METRIC VALUE BASED LOG ANOMALY DETECTION START ###############

print("Starting METRIC VALUE BASED LOG ANOMALY DETECTION")
metric_partitioning_config = FeatureExtractorConfig(
    group_by_category=log_attribute_fields  # messagetype
)

metric_feature_extractor = FeatureExtractor(metric_partitioning_config)

# Want to extract to create feature vector based on some structured attributes (which i dont want to group by), grouped by log record type/message type?
# When creating the feature vector, lib was not accepting empty log_vector ---> changed this in lib code

x, incident_metrics = metric_feature_extractor.convert_to_feature_vector(
    None, attributes=incident_attributes, timestamps=incident_timestamps
)
y, reference_metrics = metric_feature_extractor.convert_to_feature_vector(
    None, attributes=reference_attributes, timestamps=reference_timestamps
)

incident_metrics_count = metric_feature_extractor.convert_to_counter_vector(
    None, attributes=incident_attributes, timestamps=incident_timestamps
)
reference_metrics_count = metric_feature_extractor.convert_to_counter_vector(
    None, attributes=reference_attributes, timestamps=reference_timestamps
)


incident_metrics_count_msgtype = {
    x[0]: list(x[1][constants.LOGLINE_COUNTS]) for x in incident_metrics_count.groupby(log_msgtype)
}
reference_metrics_count_msgtype = {
    x[0]: list(x[1][constants.LOGLINE_COUNTS]) for x in reference_metrics_count.groupby(log_msgtype)
}

incident_metrics_msgtype = {
    x[0]: list(x[1][log_metric]) for x in incident_metrics.groupby(log_msgtype)
}
reference_metrics_msgtype = {
    x[0]: list(x[1][log_metric]) for x in reference_metrics.groupby(log_msgtype)
}

lof_params = LOFParams()
lof_params.contamination = 0.001
lof_params.n_neighbors = 20
lof_params.novelty = True
all_msgtypes = set(incident_metrics_count_msgtype).union(
    set(reference_metrics_count_msgtype)
)
ad_config = AnomalyDetectionConfig(algo_name="lof", algo_params=lof_params)
anomaly_detector = AnomalyDetector(ad_config)

num_bins_metric_value = 1000
all_metrics = []
all_metrics.extend(list(reference_metrics[log_metric]))
all_metrics.extend(list(incident_metrics[log_metric]))
all_metrics = np.array(all_metrics)
_, data_bins = np.histogram(all_metrics, bins=num_bins_metric_value)

distdiv_params = DistributionDivergenceParams(n_bins=data_bins, type=["KL", "JS"])
ad_config = AnomalyDetectionConfig(
    algo_name="distribution_divergence", algo_params=distdiv_params
)
anomaly_ranker = AnomalyDetector(ad_config)

for msgtype in all_msgtypes:
    if (
        msgtype not in incident_metrics_msgtype
        or msgtype not in reference_metrics_msgtype
    ):
        continue

    incident_metric_vals = incident_metrics_msgtype[msgtype]
    incident_metric_freqs = incident_metrics_count_msgtype[msgtype]
    incident_metric_data = pd.DataFrame(
        np.repeat(incident_metric_vals, incident_metric_freqs), columns=["data"]
    )

    reference_metric_vals = reference_metrics_msgtype[msgtype]
    reference_metric_freqs = reference_metrics_count_msgtype[msgtype]
    reference_metric_data = pd.DataFrame(
        np.repeat(reference_metric_vals, reference_metric_freqs), columns=["data"]
    )

    if len(reference_metric_data) < 2:
        continue
    anomaly_detector.fit(reference_metric_data)

    lof_prediction = anomaly_detector.predict(incident_metric_data)

    anomaly_ranker.fit(reference_metric_data)
    kl_div, js_div = anomaly_ranker.predict(incident_metric_data)

    stats = {
        "num_anomalies_metric": len(lof_prediction[lof_prediction == -1]),
        "kl_div_metric": kl_div,
        "js_div_metric": js_div,
    }
    if msgtype in stats_per_msgtype:
        stats_per_msgtype[msgtype].update(stats)
    else:
        stats_per_msgtype[msgtype] = stats

############### METRIC VALUE BASED LOG ANOMALY DETECTION END ###############

fields = [
    "num_anomalies_metric",
    "num_anomalies_freq_count",
    "kl_div_metric",
    "kl_div_freq_count",
    "js_div_metric",
    "js_div_freq_count",
]
for field in fields:
    print(field)
    for msgtype, stats in stats_per_msgtype.items():
        if field not in stats or np.isnan(stats[field]):
            stats_per_msgtype[msgtype][field] = 0.0
    for msgtype, stats in sorted(stats_per_msgtype.items(), key=lambda item: item[1][field], reverse=True)[:30]:
        print ('\tMsgType: ', msgtype, ' : ', field, ': ',stats[field])
