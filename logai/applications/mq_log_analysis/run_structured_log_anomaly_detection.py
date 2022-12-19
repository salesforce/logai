#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import os
from logai.applications.mq_log_analysis.structured_logs.logAD import (
    StructuredLogAnomalyDetection,
)
from logai.applications.mq_log_analysis.log_print_utils import (
    get_topk_most_anomalous,
    print_topk_most_anomalous,
)

if __name__ == "__main__":
    data_dir = "../../datasets/mq/"
    incident_dir = "usa26_rac_1_2022-03-07"
    log_record_type = "mqfrm"
    data_dir = os.path.join(data_dir, log_record_type, incident_dir)

    incident_filepath = os.path.join(data_dir, "incident_data.csv")
    reference_filepath = os.path.join(data_dir, "reference_data.csv")

    stats_per_msgtype = {}

    log_metrics = ["WallClockTime", "AppServerCpuTime", "DbCpuTime"]

    structlogAD = StructuredLogAnomalyDetection(
        anomaly_detection_by_metric_or_freq="freq_count"
    )
    stats_per_msgtype = structlogAD.structured_log_anomaly_detection(
        incident_filepath, reference_filepath
    )

    for log_metric in log_metrics:
        structlogAD = StructuredLogAnomalyDetection(
            anomaly_detection_by_metric_or_freq="metric", log_metric=log_metric
        )
        stats_per_msgtype = structlogAD.structured_log_anomaly_detection(
            incident_filepath, reference_filepath, stats_per_msgtype
        )

    for field in structlogAD.output_fields:
        output_analysis_freqcount = get_topk_most_anomalous(
            stats_per_msgtype, field, "freq_count", topk=5
        )
        print_topk_most_anomalous(stats_per_msgtype, field, "freq_count", topk=5)
    for log_metric in log_metrics:
        for field in structlogAD.output_fields:
            output_analysis_metric = get_topk_most_anomalous(
                stats_per_msgtype, field, log_metric, topk=5
            )
            print_topk_most_anomalous(stats_per_msgtype, field, log_metric, topk=5)
