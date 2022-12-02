#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import re 
import os 
import pandas as pd
from sklearn import config_context 
from logai.applications.application_interfaces import WorkFlowConfig
from logai.dataloader.data_loader import FileDataLoader
from logai.preprocess.preprocess import Preprocessor
from logai.utils import constants
from logai.preprocess.preprocess import PreprocessorConfig, Preprocessor
import yaml 
from datetime import datetime
from sklearn.model_selection import train_test_split

from attr import dataclass
from logai.config_interfaces import Config

@dataclass
class DatasetLogADConfig(Config):

    dataset_config: dict = {
        'log_format': "",
        'date_time_format' : '%Y-%m-%d-%H.%M.%S',
        'custom_replace_list': [],
        'parsed_date_time_format': '%Y-%m-%d %H:%M:%S',
        'logsequence_delim': '. ',
        'config_file': '',
        'label_file': None,
    }

    field_config: dict = {
        'label_field' : 'labels',
        'data_field' : 'loglines',
        'count_field' : 'count'
    }

    preprocess_config: dict = {
        'train_remove_duplicates': False, 
        'dev_remove_duplicates': False,
        'test_remove_duplicates': True,
        'output_relative_dir': 'output/nonparsed'
    }

    def from_dict(self, config_dict):
        super().from_dict(config_dict)
        return 


class DatasetLogAD:
    """
    LogAD workflow for open datasets
    """

    def __init__(self, config: DatasetLogADConfig):
        """
        Initiate Dataset Log AD with given configuration
        :param config:
        """
        self.log_format = config.dataset_config['log_format']
        self.date_time_format = config.dataset_config['date_time_format']
        self.custom_replace_list = config.dataset_config['custom_replace_list']
        self.parsed_date_time_format = config.dataset_config['parsed_date_time_format']
        self.logsequence_delim = config.dataset_config['logsequence_delim']
        self.config_file = config.dataset_config['config_file']
        self.label_file = config.dataset_config['label_file']

        self.data_field = config.field_config['data_field']
        self.label_field = config.field_config['label_field']
        self.count_field = config.field_config['count_field']

        self.train_remove_duplicates = config.preprocess_config['train_remove_duplicates']
        self.dev_remove_duplicates = config.preprocess_config['dev_remove_duplicates']
        self.test_remove_duplicates = config.preprocess_config['test_remove_duplicates']

        self.output_dir = config.preprocess_config['output_relative_dir']

    def load_data(self, filename):
        headers, regex = self.generate_logformat_regex()
        df_log = self.log_to_dataframe(filename, regex, headers)
        return df_log

    def log_to_dataframe(self, log_file, regex, headers):
        """ Function to transform log file to dataframe
        """
        log_messages = []
        linecount = 0
        cnt = 0
        with open(log_file, 'r', encoding="utf8", errors='ignore') as fin:
            for line in fin.readlines():
                cnt += 1
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
        print("Total size after encoding is", linecount, cnt)
        logdf = pd.DataFrame(log_messages, columns=headers, dtype=str)
        return logdf

    def generate_logformat_regex(self):
        """
        Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', self.log_format)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

    def preprocess_logs(self):
        # need dataset specific implementation 
        return 

    def get_log_labels(data_column: pd.Series):
        # need dataset specific implementation
        return 

    def get_log_data(self, data_loader_config):
        # need dataset specific implementation
        return 

    def generate_train_dev_test_data(self, preprocessed_results, train_file, dev_file, test_file):

        preprocessed_results_neg = preprocessed_results[preprocessed_results[self.label_field]==0]
        preprocessed_results_pos = preprocessed_results[preprocessed_results[self.label_field]==1]

        print ('Number of anomalous & non-anomalous blocks: ', len(preprocessed_results_pos), len(preprocessed_results_neg))

        train_df, test_df = train_test_split(preprocessed_results_neg, test_size=0.22)
        train_df, dev_df = train_test_split(train_df, test_size=0.1)
        test_df = pd.concat([preprocessed_results_pos, test_df])
        if self.is_logline_sequence(train_df):
            self.concat_logsequence(train_df, dev_df, test_df)

        if self.train_remove_duplicates:
            train_df = self.postprocess_remove_duplicates(train_df)
        if self.dev_remove_duplicates:
            dev_df = self.postprocess_remove_duplicates(dev_df)
        if self.test_remove_duplicates:
            test_df = self.postprocess_remove_duplicates(test_df)

        train_df.to_csv(train_file)
        test_df.to_csv(test_file)
        dev_df.to_csv(dev_file)
        return

    def postprocess_remove_duplicates(self, data_df):
        data_uniq = data_df.pivot_table(columns=[self.data_field, self.label_field], aggfunc='size')
        data_uniq_df = data_uniq.index.to_frame()
        data_uniq_df[self.count_field] = list(data_uniq)
        return data_uniq_df


    def is_logline_sequence(self, data_df):
        return type(data_df[self.data_field][0])==list

    def concat_logsequence(self, train_df, dev_df, test_df):
        test_df[self.data_field] = test_df[self.data_field].apply(lambda x: self.logsequence_delim.join(x))
        train_df[self.data_field] = train_df[self.data_field].apply(lambda x: self.logsequence_delim.join(x))
        dev_df[self.data_field] = dev_df[self.data_field].apply(lambda x: self.logsequence_delim.join(x))


    def create_dataset(self):
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), self.config_file)
        with open(CONFIG_PATH, "r") as f:
            config_yaml  = yaml.full_load(f)
            f.close()
        config = WorkFlowConfig()
        config.from_dict(config_yaml)
        self.log_file_raw = config.data_loader_config.filepath
        log_file_raw_basename = os.path.splitext(os.path.basename(self.log_file_raw))[0]
        output_dir = os.path.join(os.path.dirname(self.log_file_raw), self.output_dir)
        self.log_file = os.path.join(output_dir, log_file_raw_basename+'_processed.csv')
        config.data_loader_config.filepath = self.log_file
        train_file = os.path.join(output_dir, log_file_raw_basename+"_train.csv")
        dev_file = os.path.join(output_dir, log_file_raw_basename+"_dev.csv")
        test_file = os.path.join(output_dir, log_file_raw_basename+"_test.csv")

        if not all([os.path.exists(x) for x in [train_file, dev_file, test_file]]):
            self.preprocess_logs()
            preprocessed_result = self.get_log_data(config.data_loader_config)
            self.generate_train_dev_test_data(preprocessed_result, train_file, dev_file, test_file)