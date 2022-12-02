#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import os 
import pandas as pd
from logai.dataloader.data_loader import FileDataLoader, OpensetDataLoader, DataLoaderConfig
from logai.utils import constants
from logai.preprocess.preprocess import PreprocessorConfig, Preprocessor
from datetime import datetime
from logai.applications.public_datasets.data_preprocessing.DatasetLogAD import DatasetLogAD, DatasetLogADConfig


class BGLLogAD(DatasetLogAD):
    def __init__(self, config: DatasetLogADConfig):
        super(BGLLogAD, self).__init__(config)

    def preprocess_logs(self):
        if not os.path.exists(self.log_file):
            # TODO: integrate the data loading logic in DataLoader
            # dataloader = OpensetDataLoader(DataLoaderConfig(config.dataset_config[]))
            # log_data = dataloader.load_data(self.log_format)
            log_data = self.load_data(self.log_file_raw)
            log_data['timestamp'] = log_data['Time'].astype(str).apply(lambda x: datetime.strptime(x, self.date_time_format))
            log_data = log_data.dropna()
            log_data.to_csv(self.log_file)

    def get_log_labels(self, data_column: pd.Series):
        return data_column.apply(lambda x: 1 if x!="-" else 0)

    def get_log_data(self, data_loader_config):
        data_loader = FileDataLoader(data_loader_config)
        logrecord = data_loader.load_data()
        print ('Data loaded, number of loglines: ', len(logrecord.body[constants.LOGLINE_NAME]))
        preprocessor_config = PreprocessorConfig(custom_replace_list=self.custom_replace_list)
        preprocessor = Preprocessor(preprocessor_config)
        preprocessed_loglines, custom_patterns = preprocessor.clean_log(logrecord.body[constants.LOGLINE_NAME])
        print ('Data cleaned, number of loglines: ', len(preprocessed_loglines))
        preprocessed_logtimestamps = logrecord.attributes['timestamp']#.apply(lambda x: datetime.strptime(x, parsed_date_time_format))
        preprocessed_labels = self.get_log_labels(logrecord.attributes['Label'])
        preprocessed_results = pd.DataFrame({'loglines': preprocessed_loglines, 'labels': preprocessed_labels, 'timestamps': preprocessed_logtimestamps})
        return preprocessed_results


if __name__ == "__main__":
    config = DatasetLogADConfig()
    config.dataset_config['log_format'] = '<Label> <Id> <Date> <Code1> <Time> <Code2> <Content>'
    config.dataset_config['date_time_format'] = '%Y-%m-%d-%H.%M.%S.%f'
    config.dataset_config['custom_replace_list'] = [
                    [r'(0x)[0-9a-fA-F]+', '[HEX]'],
                    [r'\d+.\d+.\d+.\d+', '[IP]'],
                    [r'\d+', '[INT]']
                ]
    config.dataset_config['parsed_date_time_format'] = '%Y-%m-%d %H:%M:%S.%f'
    config.dataset_config['logsequence_delim'] = None
    config.dataset_config['config_file'] = "bgl/bgl_config_parsefree.yaml"

    bgl_logAD = BGLLogAD(config=config)
    bgl_logAD.create_dataset()

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

        def is_logline_sequence(data_df):
        return type(data_df['loglines'][0])==list

    def concat_logsequence(train_df, dev_df, test_df):
        test_df['loglines'] = test_df['loglines'].apply(lambda x: logsequence_delim.join(x))
        train_df['loglines'] = train_df['loglines'].apply(lambda x: logsequence_delim.join(x))
        dev_df['loglines'] = dev_df['loglines'].apply(lambda x: logsequence_delim.join(x))

    def generate_train_dev_test_data(preprocessed_results, train_file, dev_file, test_file):

        preprocessed_results_neg = preprocessed_results[preprocessed_results['labels']==0]
        preprocessed_results_pos = preprocessed_results[preprocessed_results['labels']==1]

        print ('Number of anomalous & non-anomalous blocks: ', len(preprocessed_results_pos), len(preprocessed_results_neg))

        train_df, test_df = train_test_split(preprocessed_results_neg, test_size=0.22)
        train_df, dev_df = train_test_split(train_df, test_size=0.1)
        test_df = pd.concat([preprocessed_results_pos, test_df])
        if is_logline_sequence(train_df):
            concat_logsequence(train_df, dev_df, test_df)

        train_df = postprocess_remove_duplicates(train_df)
        dev_df = postprocess_remove_duplicates(dev_df)
        test_df = postprocess_remove_duplicates(test_df)

        train_df.to_csv(train_file)
        test_df.to_csv(test_file)
        dev_df.to_csv(dev_file)
        return

    def postprocess_remove_duplicates(data_df):
        data_uniq = data_df.pivot_table(columns=['loglines', 'labels'], aggfunc='size')
        data_uniq_df = data_uniq.index.to_frame()
        data_uniq_df['count'] = list(data_uniq)
        return data_uniq_df
    

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
label_file = None
train_file = os.path.join(output_dir, log_file_raw_basename+"_train.csv")
dev_file = os.path.join(output_dir, log_file_raw_basename+"_dev.csv")
test_file = os.path.join(output_dir, log_file_raw_basename+"_test.csv")

if not all([os.path.exists(x) for x in [train_file, dev_file, test_file]]):
    preprocess_logs(log_file, log_file_raw)
    preprocessed_result = get_log_data(config, label_file)
    generate_train_dev_test_data(preprocessed_result, train_file, dev_file, test_file)
    '''
