#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#

import os
import pandas as pd
from logai.applications.mq_log_analysis.unstructured_logs.logClustering import (
    LogClustering,
    LogClusteringPerFunction,
)


def collate_all_files(data_dir, filepath):
    data = None
    for file in os.listdir(data_dir):
        data_i = pd.read_csv(os.path.join(data_dir, file))
        if data is None:
            data = data_i
        else:
            data = pd.concat([data, data_i])
    if filepath is not None:
        data.to_csv(filepath)
    else:
        return data


if __name__ == "__main__":
    data_dir = "../../datasets/mq/"
    # incident_dir = 'na163_2022-03-04/reference_data'
    # incident_dir = 'cs142_2022-03-09'
    # incident_dir = 'cs191_2022-03-17'
    # incident_dir = 'cs218_2022-03-20'
    # incident_dir = 'na94_2022-03-23'
    # incident_dir = 'cs5_2022-03-26'
    # incident_dir = 'na100_2022-03-29'
    incident_dir = "na128_2022-03-29"
    # incident_dir = 'um8_2022-03-10/ref1'
    log_record_type = "mqdbg"
    data_dir = os.path.join(data_dir, log_record_type, incident_dir)

    filepath = os.path.join(data_dir, "incident_data.csv")
    if not os.path.exists(filepath):
        collate_all_files(data_dir, filepath)

    num_clusters = input("Enter number of clusters (default=5):")
    if num_clusters == "":
        num_clusters = 5
    else:
        num_clusters = int(num_clusters)

    logClustering = LogClustering(num_clusters=num_clusters)
    clusterwise_template_param_data = logClustering.log_clustering(filepath)
    logClustering.print_cluster_interactive(clusterwise_template_param_data)

    logClustering = LogClusteringPerFunction()
    logClustering.log_clustering(filepath)
