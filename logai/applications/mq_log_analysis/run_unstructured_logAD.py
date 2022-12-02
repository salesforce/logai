#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#

import os 
import pandas as pd 
from logai.applications.mq_log_analysis.unstructured_logs.logAD import UnstructuredLogAnomalyDetection
from logai.applications.mq_log_analysis.log_print_utils import *

def collate_all_files(data_dir, filepath=None):
    data = None 
    for file in os.listdir(data_dir):
        data_i = pd.read_csv(os.path.join(data_dir, file))
        if data is None:
            data = data_i 
        else:
            data = pd.concat([data, data_i])
    print ('Collated all data')
    if filepath:
        data.to_csv(filepath)
    else:
        return data 
        
if __name__=="__main__":
    data_dir = '../../datasets/mq/'
    incident_dir = 'na163_2022-03-04'
    log_record_type = 'mqdbg'
    data_dir = os.path.join(data_dir, log_record_type, incident_dir)

    incident_dir = os.path.join(data_dir, 'incident_data')
    incident_filepath = os.path.join(incident_dir, 'incident_data.csv')
    parsed_output_filepath = incident_filepath.replace('.csv', '_parsed.csv')
    
    reference_dir = os.path.join(data_dir, 'reference_data')
    reference_filepath = os.path.join(reference_dir, 'reference_data.csv')
    parsed_output_filepath = reference_filepath.replace('.csv', '_parsed.csv')

    if not os.path.exists(incident_filepath):
        collate_all_files(incident_dir, incident_filepath)

    if not os.path.exists(reference_filepath):
        collate_all_files(reference_dir, reference_filepath)

    unstructlogAD = UnstructuredLogAnomalyDetection()

    stats_per_msgtype = unstructlogAD.unstructured_log_anomaly_detection(incident_filepath, reference_filepath)
    
    for field in unstructlogAD.output_fields:
        print_topk_most_anomalous(stats_per_msgtype, field, 'freq_count', topk=5)

    #parsed_result[constants.PARSED_LOGLINE_NAME] = parsed_result[constants.PARSED_LOGLINE_NAME].apply(lambda x: ' '.join(x.split(' ')[1:]))
    #print ('incident_logrecord.timestamp: ',incident_logrecord.timestamp, type(incident_logrecord.timestamp))
    
