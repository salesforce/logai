#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import os 
import pandas as pd
from logai.applications.public_datasets.data_preprocessing.DatasetLogAD import DatasetLogAD, DatasetLogADConfig 
from logai.dataloader.data_loader import FileDataLoader
from logai.information_extraction.feature_extractor import FeatureExtractorConfig, FeatureExtractor
from logai.preprocess.preprocess import Preprocessor
from logai.utils import constants
from logai.preprocess.preprocess import PreprocessorConfig, Preprocessor
from datetime import datetime


class HDFSLogAD(DatasetLogAD):

    def __init__(self, config: DatasetLogADConfig):
        super(HDFSLogAD, self).__init__(config)

    def preprocess_logs(self):
        if not os.path.exists(self.log_file):
            log_data = self.load_data(self.log_file_raw)
            log_data['timestamp'] = (log_data['Date'].astype(str) + " " + log_data['Time'].astype(str)).apply(lambda x: datetime.strptime(x, self.date_time_format))
            log_data = log_data.dropna()
            log_data.to_csv(self.log_file)

    def get_log_labels(self, data_column: pd.Series):
        blk_df = pd.read_csv(self.label_file, header=0)
        anomaly_blk = set(blk_df[blk_df['Label']=='Anomaly']['BlockId'])
        return data_column.apply(lambda x: 1 if x in anomaly_blk else 0)

    def get_log_data(self, data_loader_config):
        data_loader = FileDataLoader(data_loader_config)
        logrecord = data_loader.load_data()
        print ('Data loaded, number of loglines: ', len(logrecord.body[constants.LOGLINE_NAME]))
        preprocessor_config = PreprocessorConfig(custom_replace_list=self.custom_replace_list)
        preprocessor = Preprocessor(preprocessor_config)
        preprocessed_loglines, custom_patterns = preprocessor.clean_log(logrecord.body[constants.LOGLINE_NAME])
        print ('Data cleaned, number of loglines: ', len(preprocessed_loglines))
        preprocessed_logtimestamps = logrecord.attributes['timestamp']#.apply(lambda x: datetime.strptime(x, parsed_date_time_format))
        block_ids = custom_patterns['[BLOCK]']
        block_ids = block_ids.apply(lambda x: ', '.join(set(x)))
        preprocessed_results = pd.DataFrame({'loglines': preprocessed_loglines, 'block_id': block_ids, 'timestamps': preprocessed_logtimestamps})
        
        fe_config = FeatureExtractorConfig(group_by_category=['block_id'])
        feature_extractor = FeatureExtractor(fe_config)
        preprocessed_logsequence, _ = feature_extractor.convert_to_sequence(log_pattern=preprocessed_results['loglines'], attributes=preprocessed_results['block_id'], timestamps=preprocessed_results['timestamps'])
        preprocessed_logsequence['labels'] = self.get_log_labels(preprocessed_logsequence['block_id'])

        return preprocessed_logsequence

if __name__=="__main__":

    config = DatasetLogADConfig()
    config.dataset_config['log_format'] = '<Date> <Time> <Pid> <Level> <Content>'
    config.dataset_config['date_time_format'] = '%y%m%d %H%M%S'
    config.dataset_config['custom_replace_list'] = [
                    [r'(blk_-?\d+)', '[BLOCK]'],
                    [r'/?\d+\.\d+\.\d+\.\d+',  '[IP]'],
                    [r'/?(/[-\w]+)+', '[FILE]'],
                    [r'\d+', '[INT]']
                ]
    config.dataset_config['parsed_date_time_format'] = '%Y-%m-%d %H:%M:%S'
    config.dataset_config['logsequence_delim'] = '. ' # ' [SEP] '
    config.dataset_config['config_file'] = "hdfs/hdfs_config_parsefree.yaml"
    config.dataset_config['label_file'] = "../../datasets/public/HDFS/anomaly_label.csv"

    hdfs_logAD = HDFSLogAD(config=config)
    hdfs_logAD.create_dataset()
    
    
'''


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
    with open(log_file, 'r', encoding="utf8", errors='ignore') as fin:
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


CONFIG_PATH = os.path.join(os.path.dirname(__file__), config_file)
with open(CONFIG_PATH, "r") as f:
    config_yaml  = yaml.full_load(f)
    f.close()
config = WorkFlowConfig()
config.from_dict(config_yaml)
log_file_raw = config.data_loader_config.filepath
log_file_raw_basename = os.path.splitext(os.path.basename(log_file_raw))[0]
output_dir = os.path.join(os.path.dirname(log_file_raw), 'output/nonparsed')
log_file = os.path.join(output_dir, log_file_raw_basename+'_processed.csv')
config.data_loader_config.filepath = log_file
label_file = os.path.join(os.path.dirname(log_file_raw), "anomaly_label.csv")
train_file = os.path.join(output_dir, log_file_raw_basename+"_train.csv")
dev_file = os.path.join(output_dir, log_file_raw_basename+"_dev.csv")
test_file = os.path.join(output_dir, log_file_raw_basename+"_test.csv")

if not all([os.path.exists(x) for x in [train_file, dev_file, test_file]]):
    preprocess_logs(log_file, log_file_raw)
    preprocessed_result = get_log_data(config, label_file)
    generate_train_dev_test_data(preprocessed_result, train_file, dev_file, test_file)
'''


