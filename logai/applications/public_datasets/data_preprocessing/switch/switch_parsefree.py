#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import logging
import re 
import os 
import pandas as pd
import pickle as pkl
from sklearn import config_context 
from logai.applications.application_interfaces import WorkFlowConfig
from logai.dataloader.data_loader import DataLoaderConfig, FileDataLoader
from logai.information_extraction.feature_extractor import FeatureExtractorConfig, FeatureExtractor
from logai.preprocess.preprocess import Preprocessor
from logai.dataloader.data_model import LogRecordObject
from logai.information_extraction.log_parser import LogParser
from logai.utils import constants
from logai.preprocess.preprocess import PreprocessorConfig, Preprocessor
import yaml 
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

max_failure_pred_time = 30 * 60 
failure_pred_duration = 24 * 60 * 60 

def preprocess_labels(label_file):
    with open(label_file) as fin:
        labels = []
        for line in fin.readlines():
            line = [int(x) for x in line.strip().split(' ')]
            failure_template_id = line[0]
            failure_timestamp = line[1]
            failure_pred_end = line[1] - max_failure_pred_time
            failure_pred_start = failure_pred_end - failure_pred_duration


print (os.path.dirname(__file__))
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "switch_config_parsefree.yaml")
print (CONFIG_PATH)
with open(CONFIG_PATH, "r") as f:
    config_yaml  = yaml.full_load(f)
    f.close()
config = WorkFlowConfig()
config.from_dict(config_yaml)
log_file_raw = config.data_loader_config.filepath
log_file_raw_basename = os.path.splitext(os.path.basename(log_file_raw))[0]
output_dir = os.path.join(os.path.dirname(log_file_raw), 'output/nonparsed')
label_file = os.path.join(os.path.dirname(log_file_raw), 'failure_info.txt')
preprocess_labels(label_file)
log_file = os.path.join(output_dir, log_file_raw_basename+'_processed.csv')
config.data_loader_config.filepath = log_file
train_file = os.path.join(output_dir, log_file_raw_basename+"_train.csv")
dev_file = os.path.join(output_dir, log_file_raw_basename+"_dev.csv")
test_file = os.path.join(output_dir, log_file_raw_basename+"_test.csv")

preprocess_logs(config, log_file, log_file_raw)
preprocessed_result = get_log_sequence()
train_df, dev_df, test_df = generate_train_dev_test_data(preprocessed_result, train_file, dev_file, test_file)



