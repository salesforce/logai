import os
from _csv import writer
import argparse
import sys
from logai.algorithms.factory import AlgorithmFactory
from logai.algorithms.nn_model.logbert.configs import LogBERTConfig
from logai.algorithms.vectorization_algo.logbert import LogBERTVectorizerParams
from logai.applications.openset.anomaly_detection.openset_anomaly_detection_workflow import OpenSetADWorkflowConfig, validate_config_dict
from logai.dataloader.data_model import LogRecordObject
from logai.dataloader.parse_logs import ParseLogs
from logai.preprocess.mask_logs import MaskLogLine
from logai.utils.file_utils import read_file
from logai.utils.dataset_utils import split_train_dev_test_for_anomaly_detection
import logging
import pandas as pd
from logai.dataloader.data_loader import FileDataLoader
from logai.preprocess.hdfs_preprocessor import HDFSPreprocessor
from logai.information_extraction.log_parser import LogParser
from logai.preprocess.openset_partitioner import OpenSetPartitioner
from logai.analysis.nn_anomaly_detector import NNAnomalyDetector
from logai.information_extraction.log_vectorizer import LogVectorizer
from logai.utils import constants
from datasets import Dataset as HFDataset
import pandas as pd


# os.environ["CUDA_VISIBLE_DEVICES"] = ""

parser = argparse.ArgumentParser(description="Set some values")
parser.add_argument("--config_file", type=str, required=True, help="The value is path of your config file")
parser.add_argument("--predict_only", type=str, required=True, help="The value is 1 if predict only, 2 if both training and prediction")
parser.add_argument("--process_data", type=str, required=True, help="The value is 1 if processing needs else 2")
parser.add_argument("--partition_data", type=str, required=True, help="The value is 1 if partition needs else 2")
parser.add_argument("--split_needs", type=str, required=True, help="The value is 1 if split needs else 2")
parser.add_argument("--tokenization_needed", type=str, required=True, help="The value is 1 if tokenization_needed needs else 2")
args = parser.parse_args()


config_path = args.config_file
predict_only = int(args.predict_only)
process_data = int(args.process_data)
partition_data = int(args.partition_data)
split_needs = int(args.split_needs)
tokenization_needed = int(args.tokenization_needed)


config_parsed = read_file(config_path)
config_dict = config_parsed["workflow_config"]
config = OpenSetADWorkflowConfig.from_dict(config_dict)

processed_filepath = os.path.join(config.output_dir, 'processed.csv')

partitioned_filepath = os.path.join(config.output_dir, 'partitioned.csv')

train_filepath = os.path.join(config.output_dir, 'train_ml.csv')
dev_filepath = os.path.join(config.output_dir, 'dev_ml.csv')
test_filepath = os.path.join(config.output_dir, 'test_ml.csv')


dev_tokenizer_path = os.path.join(config.output_dir, "dev_tokenizer_sm")
train_tokenizer_path = os.path.join(config.output_dir, "train_tokenizer_sm")
test_tokenizer_path = os.path.join(config.output_dir, "test_tokenizer_sm")

vectorizer = None

def get_vectorizer():
  global vectorizer
  if vectorizer is None:
    vectorizer = LogVectorizer(config.log_vectorizer_config)
    train_data = LogRecordObject.load_from_csv(train_filepath)
    vectorizer.fit(train_data)
  return vectorizer


def get_features(tokenizer_path, data_path=None, data=None, tokenization_needed=1):
  if tokenization_needed == 1:
    if data:
      features = get_vectorizer().transform(data)
      features.save_to_disk(tokenizer_path)
    elif data_path:
      data = LogRecordObject.load_from_csv(data_path)
      features = get_vectorizer().transform(data)
      features.save_to_disk(tokenizer_path)
    else:
      return None
    del data
  else:
    if os.path.exists(tokenizer_path):
      features = HFDataset.load_from_disk(tokenizer_path)
    else:
      return None
  print("Vectorization complete for %s" % tokenizer_path)
  return features


def mask_logs(logline):
  masker = MaskLogLine(logline)
  nlogline = masker.mask_log_line_efficient_way(logline)
  return nlogline


def process_df(df, label):
  df["epoch"] = df["epoch"].astype(int)
  has_zero_or_na_column = (df['epoch'] == 0) | df['epoch'].isna()
  if has_zero_or_na_column.any():
    dataset = "train" if label == 0 else "test"
    print("Dataset %s has corrupt epochs" % dataset)
    exit(1)
  df = df.dropna(subset=['pid'])
  print(df.shape)
  #df.drop_duplicates(subset='body', keep='first', inplace=True)
  #print("after duplicate removal")
  #print(df.shape)
  #num_input = input("If this looks fine, press 1, else press 2 for exit, press 3 for sampling")
  # num = 3 # int(num_input)
  # if num == 2:
  #     exit(1)
  # if num == 3:
  #     #n_ = input("Enter number by which you want to divide")
  #     #n_ = int(n_)
  #     df = divide_df(df, 100)
  #     print("after division")
  #     print(df.shape)
  df["source_file"] = df["source_file"].str.replace(r'\..*', '', regex=True)
  _SEVERITY_MAP = {
    "DEBUG": 0,
    "INFO": 1,
    "WARNING": 2,
    "ERROR": 3,
    "CRITICAL": 4,
    "FATAL": 5,
  }
  df = df.rename(
    columns={"body": constants.LOGLINE_NAME, "level": constants.LOG_LEVEL}
  )
  df[constants.LOG_SEVERITY_NUMBER] = df[constants.LOG_LEVEL].map(_SEVERITY_MAP)
  df[constants.LOG_SEVERITY_NUMBER] = df[constants.LOG_SEVERITY_NUMBER].fillna(1)
  df[constants.LOG_TIMESTAMPS] = pd.to_datetime(df["epoch"], unit='s')
  # df[constants.LOGLINE_NAME] = df[constants.LOGLINE_NAME].apply(mask_logs)
  # print("Masking of logs is complete")
  df[constants.SPAN_ID] = df["pid"] #df["source_file"] + "_" + df[constants.LOG_SEVERITY_NUMBER].astype(str) # df["pid"] + "_" + + logs_df["hh"].astype(str)
  df[constants.LABELS] = label
  return df

def divide_df(df, n):
    num_rows_to_select = len(df) // n
    selected_data = df.sample(n=num_rows_to_select)
    return selected_data


if process_data == 1:

  data_types = {'Column1':str, 'Column2': str, 'Column3': str, 'Column4': int, 'Column5': str, 'Column6': int, 'Column7': str}

  if config.data_loader_config.test and os.path.exists(config.data_loader_config.test):
    test_df = pd.read_csv(config.data_loader_config.test, index_col=0, dtype=data_types)#, nrows=20000)
  else:
    exit(1)
  if config.data_loader_config.train and os.path.exists(config.data_loader_config.train):
    train_df = pd.read_csv(config.data_loader_config.train, index_col=0, dtype=data_types, encoding='unicode_escape')#, nrows=200000)
  else:
    exit(1)

  train_df = process_df(train_df, 0)
  test_df = process_df(test_df, 1)


  # Perform vertical concatenation
  logs_df = pd.concat([train_df, test_df])

  # Reset the index
  logs_df = logs_df.reset_index(drop=True)


  metadata = {constants.LOG_TIMESTAMPS: [constants.LOG_TIMESTAMPS], constants.SPAN_ID: [constants.SPAN_ID],
              "body": [constants.LOGLINE_NAME, "source_file"], constants.LABELS: [constants.LABELS],
              "severity_text": ["log_level"], constants.LOG_SEVERITY_NUMBER: [constants.LOG_SEVERITY_NUMBER]}

  logrecord = LogRecordObject.from_dataframe(logs_df, metadata)

  logrecord.save_to_csv(processed_filepath)
else:
  logrecord = LogRecordObject.load_from_csv(processed_filepath)
  logrecord.timestamp[constants.LOG_TIMESTAMPS] = pd.to_datetime(logrecord.timestamp[constants.LOG_TIMESTAMPS] )

print (logrecord.body[constants.LOGLINE_NAME])

if partition_data == 1:
  partitioner = OpenSetPartitioner(config.open_set_partitioner_config)
  logrecord = partitioner.partition(logrecord)
  logrecord.save_to_csv(partitioned_filepath)
else:
  logrecord = LogRecordObject.load_from_csv(partitioned_filepath)

print (logrecord.body[constants.LOGLINE_NAME])

if split_needs == 1:
  train_data, dev_data, test_data = split_train_dev_test_for_anomaly_detection(
                  logrecord,training_type=config.training_type,
                  test_data_frac_neg_class=config.test_data_frac_neg,
                  test_data_frac_pos_class=config.test_data_frac_pos,
                  shuffle=config.train_test_shuffle
              )

  train_data.save_to_csv(train_filepath)
  dev_data.save_to_csv(dev_filepath)
  test_data.save_to_csv(test_filepath)
else:
  train_data, dev_data, test_data = LogRecordObject.load_from_csv(train_filepath), LogRecordObject.load_from_csv(dev_filepath), LogRecordObject.load_from_csv(test_filepath)

print ('Train/Dev/Test Anomalous', len(train_data.labels[train_data.labels[constants.LABELS]==1]),
                                   len(dev_data.labels[dev_data.labels[constants.LABELS]==1]),
                                   len(test_data.labels[test_data.labels[constants.LABELS]==1]))
print ('Train/Dev/Test Normal', len(train_data.labels[train_data.labels[constants.LABELS]==0]),
                                   len(dev_data.labels[dev_data.labels[constants.LABELS]==0]),
                                   len(test_data.labels[test_data.labels[constants.LABELS]==0]))

dev_features = get_features(dev_tokenizer_path, dev_filepath, dev_data, tokenization_needed)
train_features = get_features(train_tokenizer_path, train_filepath, train_data, tokenization_needed)


anomaly_detector = NNAnomalyDetector(config=config.nn_anomaly_detection_config)
if predict_only == 2:
  anomaly_detector.fit(train_features, dev_features)

del train_features, dev_features

test_features = get_features(test_tokenizer_path, test_filepath, test_data, tokenization_needed)
predict_results = anomaly_detector.predict(test_features)
print(predict_results)



