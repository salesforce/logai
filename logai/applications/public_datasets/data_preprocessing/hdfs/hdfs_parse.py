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
from logai.information_extraction.categorical_encoder import CategoricalEncoder, CategoricalEncoderConfig
from datetime import datetime
from sklearn.model_selection import train_test_split
from logai.algorithms.nn_model.transformers import TransformerAlgoConfig, TransformerAlgo

def load_data(filename, log_format):
    headers, regex = generate_logformat_regex(log_format)
    df_log = log_to_dataframe(filename, regex, headers)
    return df_log


def log_to_dataframe(log_file, regex, headers):
    """ Function to transform log file to dataframe
    """
    log_messages = []
    linecount = 0
    cnt = 0
    with open(log_file, 'r') as fin:
        for line in fin.readlines():
            cnt += 1
            try:
                match = regex.search(line.strip())
                message = [match.group(header) for header in headers]
                log_messages.append(message)
                linecount += 1
            except Exception as e:
                # print("\n", line)
                # print(e)
                pass
    print("Total size after encoding is", linecount, cnt)
    logdf = pd.DataFrame(log_messages, columns=headers, dtype=str)
    logdf.insert(0, 'LineId', None)
    logdf['LineId'] = [i + 1 for i in range(linecount)]
    return logdf

def generate_logformat_regex(logformat):
    """ Function to generate regular expression to split log messages
    """
    headers = []
    splitters = re.split(r'(<[^<>]+>)', logformat)
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


def preprocess_logs(log_file, log_file_raw):
    if not os.path.exists(log_file):
        log_format = '<Date> <Time> <Pid> <Level> <Content>'
        date_time_format = '%y%m%d %H%M%S'
        
        log_data = load_data(log_file_raw, log_format)
        log_data['timestamp'] = (log_data['Date'].astype(str) + " " + log_data['Time'].astype(str)).apply(lambda x: datetime.strptime(x, date_time_format))
        log_data = log_data.dropna()
        log_data.to_csv(log_file)
        
def parse_logs(log_file, log_file_parsed):
    if not os.path.exists(log_file_parsed):
        preprocess_logs(log_file)
    
        preprocessor_config = PreprocessorConfig(
            custom_replace_list=[
                [r'(blk_-?\d+)', '[BLOCK]'],
                [r'/?\d+\.\d+\.\d+\.\d+',  '[IP]'],
                [r'/?(/[-\w]+)+', '[FILE]'],
                [r'\d+', '[INT]']
            ]
        )
        data_loader = FileDataLoader(config.data_loader_config)
        logrecord = data_loader.load_data()
        print ('Data loaded, number of loglines: ', len(logrecord.body[constants.LOGLINE_NAME]))
        preprocessor = Preprocessor(preprocessor_config)
        preprocessed_loglines, custom_patterns = preprocessor.clean_log(logrecord.body[constants.LOGLINE_NAME])
        #preprocessed_logtimestamps, _ = preprocessor.clean_log(logrecord.attributes['timestamp'])
        print ('Data cleaned, number of loglines: ', len(preprocessed_loglines))

        print ("custom_patterns ", custom_patterns)
        parser = LogParser(config.log_parser_config)
        parsed_result = parser.parse(preprocessed_loglines)
        parsed_result['timestamps'] = logrecord.attributes['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        parsed_result['block_id'] = custom_patterns['[BLOCK]']
        parsed_result['block_id'] = parsed_result['block_id'].apply(lambda x: ', '.join(set(x)))
        encoder_config = CategoricalEncoderConfig()

        cat_encoder = CategoricalEncoder(encoder_config)
        parsed_result['event_id'] = cat_encoder.fit_transform(parsed_result[[constants.PARSED_LOGLINE_NAME]])

        parsed_result.to_csv(log_file_parsed)
    else:
        parsed_result = pd.read_csv(log_file_parsed)
    return parsed_result

def get_event_sequence(parsed_result, label_file):
    event_id_template_df = parsed_result[['parsed_logline', 'event_id']].drop_duplicates()
    event_id_template_map = dict(zip(event_id_template_df.event_id, event_id_template_df.parsed_logline))

    config = FeatureExtractorConfig(group_by_category=['block_id'], group_by_time='1min')
    feature_extractor = FeatureExtractor(config)
    event_sequence, _ = feature_extractor.convert_to_sequence(log_pattern=parsed_result['event_id'], attributes=parsed_result['block_id'], timestamps=parsed_result['timestamps'])

    blk_df = pd.read_csv(label_file, header=0)
    anomaly_blk = set(blk_df[blk_df['Label']=='Anomaly']['BlockId'])

    event_sequence['labels'] = event_sequence['block_id'].apply(lambda x: 1 if x in anomaly_blk else 0)
    print ('event_sequence: ',event_sequence, '\nlen of event_sequence: ', len(event_sequence))
    return event_sequence, event_id_template_map

def generate_train_test_data(event_sequence, event_id_template_map, train_file, dev_file, test_file):

    
    neg_df = event_sequence[event_sequence['labels']==0]
    pos_df = event_sequence[event_sequence['labels']==1]

    print ('Number of anomalous & non-anomalous blocks: ', len(pos_df), len(neg_df))

    train_df, test_df = train_test_split(neg_df, test_size=0.22)
    train_df, dev_df = train_test_split(train_df, test_size=0.1)
    test_df = pd.concat([pos_df, test_df])

    print ('event_id_template_map: ', event_id_template_map.keys())

    logline_delim = '. ' # ' [SEP] '
    test_df['loglines'] = test_df['event_id'].apply(lambda x: logline_delim.join([event_id_template_map[xi] for xi in x]))
    train_df['loglines'] = train_df['event_id'].apply(lambda x: logline_delim.join([event_id_template_map[xi] for xi in x]))
    dev_df['loglines'] = dev_df['event_id'].apply(lambda x: logline_delim.join([event_id_template_map[xi] for xi in x]))
    train_df.to_csv(train_file)
    test_df.to_csv(test_file)
    dev_df.to_csv(dev_file)

    train_file_txt = train_file.replace('.csv', '_text.csv')
    test_file_txt = test_file.replace('.csv', '_text.csv')
    dev_file_txt = dev_file.replace('.csv', '_text.csv')
    with open(train_file_txt, 'w') as fw:
        for l in list(train_df['loglines']):
            fw.write(l.strip()+'\n')
        
    with open(test_file_txt, 'w') as fw:
        for l in list(test_df['loglines']):
            fw.write(l.strip()+'\n')

    with open(dev_file_txt, 'w') as fw:
        for l in list(dev_df['loglines']):
            fw.write(l.strip()+'\n')

    return train_df, dev_df, test_df



CONFIG_PATH = os.path.join(os.path.dirname(__file__), "hdfs_config_parse.yaml")
with open(CONFIG_PATH, "r") as f:
    config_yaml  = yaml.full_load(f)
    f.close()
config = WorkFlowConfig()
config.from_dict(config_yaml)
log_file_raw = config.data_loader_config.filepath
log_file_raw_basename = os.path.splitext(os.path.basename(log_file_raw))[0]
output_dir = os.path.join(os.path.dirname(log_file_raw), 'output/parsed')
log_file = os.path.join(output_dir, log_file_raw_basename+'_processed.csv')
log_file_parsed = os.path.join(output_dir, log_file_raw_basename+'_parsed.csv')
config.data_loader_config.filepath = log_file
label_file = os.path.join(os.path.dirname(log_file_raw), "anomaly_label.csv")
train_file = os.path.join(output_dir, log_file_raw_basename+"_train.csv")
dev_file = os.path.join(output_dir, log_file_raw_basename+"_dev.csv")
test_file = os.path.join(output_dir, log_file_raw_basename+"_test.csv")

preprocess_logs(log_file, log_file_raw)
parsed_result = parse_logs(log_file, log_file_parsed)
event_sequence, event_id_template_map = get_event_sequence(parsed_result, label_file)
train_df, dev_df, test_df = generate_train_test_data(event_sequence, event_id_template_map, train_file, dev_file, test_file)





