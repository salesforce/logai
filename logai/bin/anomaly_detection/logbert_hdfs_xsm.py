import argparse
import os

from logai.algorithms.factory import AlgorithmFactory
from logai.algorithms.nn_model.logbert.configs import LogBERTConfig
from logai.algorithms.vectorization_algo.logbert import LogBERTVectorizerParams
from logai.applications.openset.anomaly_detection.openset_anomaly_detection_workflow import OpenSetADWorkflowConfig, validate_config_dict
from logai.dataloader.data_model import LogRecordObject
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

import torch
torch.cuda.empty_cache()

# Set the max_memory_split_size in bytes (1 GB in this example)
max_memory_split_size_bytes = 2 * 1024 * 1024 * 1024

# Set the configuration option
torch.cuda.memory.init_max_split_size = max_memory_split_size_bytes

vectorizer = None

def get_vectorizer():
  global vectorizer
  if vectorizer is None:
    vectorizer = LogVectorizer(config.log_vectorizer_config)
    train_data = LogRecordObject.load_from_csv(train_filepath)
    vectorizer.fit(train_data)
  return vectorizer


def get_features(tokenizer_path, data_path, data=None):
  if os.path.exists(tokenizer_path):
    features = HFDataset.load_from_disk(tokenizer_path)
  else:
    if not data:
      data = LogRecordObject.load_from_csv(data_path)
    features = get_vectorizer().transform(data)
    features.save_to_disk(tokenizer_path)
    del data
  print("Vectorization complete for %s" % tokenizer_path)
  return features


parser = argparse.ArgumentParser(description="Set some values")
parser.add_argument("config_file", type=str, help="The value is path of your config file")
parser.add_argument("predict_only", type=str, help="Use 1 for predict_only, 2 for training/prediction")

args = parser.parse_args()


config_path = args.config_file
predict_only= args.predict_only

print(predict_only)

config_parsed = read_file(config_path)
config_dict = config_parsed["workflow_config"]
config = OpenSetADWorkflowConfig.from_dict(config_dict)

dev_tokenizer_path = os.path.join(config.output_dir, "dev_tokenizer")
train_tokenizer_path = os.path.join(config.output_dir, "train_tokenizer")
test_tokenizer_path = os.path.join(config.output_dir, "test_tokenizer")
train_filepath = os.path.join(config.output_dir, 'train.csv')
dev_filepath = os.path.join(config.output_dir, 'dev.csv')
test_filepath = os.path.join(config.output_dir, 'test.csv')
partitioned_filepath = os.path.join(config.output_dir, 'partitioned_session.csv')
loaded_filepath = os.path.join(config.output_dir, 'loaded.csv')
preprocessed_filepath = os.path.join(config.output_dir, 'processed.csv')


if not os.path.exists(dev_tokenizer_path):

  dataloader = FileDataLoader(config.data_loader_config)
  logrecord = dataloader.load_data()
  logrecord.save_to_csv(loaded_filepath)
  print (logrecord.body[constants.LOGLINE_NAME])

  preprocessor = HDFSPreprocessor(config.preprocessor_config, config.label_filepath)
  logrecord = preprocessor.clean_log(logrecord)
  logrecord.save_to_csv(preprocessed_filepath)
  print (logrecord.body[constants.LOGLINE_NAME])

  #####
  partitioner = OpenSetPartitioner(config.open_set_partitioner_config)
  logrecord = partitioner.partition(logrecord)
  logrecord.save_to_csv(partitioned_filepath)
  print (logrecord.body[constants.LOGLINE_NAME])

  train_data, dev_data, test_data = split_train_dev_test_for_anomaly_detection(
                  logrecord,training_type=config.training_type,
                  test_data_frac_neg_class=config.test_data_frac_neg,
                  test_data_frac_pos_class=config.test_data_frac_pos,
                  shuffle=config.train_test_shuffle
              )

  train_data.save_to_csv(train_filepath)
  dev_data.save_to_csv(dev_filepath)
  test_data.save_to_csv(test_filepath)
  print ('Train/Dev/Test Anomalous', len(train_data.labels[train_data.labels[constants.LABELS]==1]),
                                     len(dev_data.labels[dev_data.labels[constants.LABELS]==1]),
                                     len(test_data.labels[test_data.labels[constants.LABELS]==1]))
  print ('Train/Dev/Test Normal', len(train_data.labels[train_data.labels[constants.LABELS]==0]),
                                     len(dev_data.labels[dev_data.labels[constants.LABELS]==0]),
                                     len(test_data.labels[test_data.labels[constants.LABELS]==0]))


if not hasattr(globals(), 'dev_data'):
    dev_data, train_data = None, None
    print("making train data as none")

if int(predict_only) == 2:
  dev_features = get_features(dev_tokenizer_path, dev_filepath, dev_data)
  train_features = get_features(train_tokenizer_path, train_filepath, train_data)

anomaly_detector = NNAnomalyDetector(config=config.nn_anomaly_detection_config)
if int(predict_only) == 2:
  anomaly_detector.fit(train_features, dev_features)
  del train_features, dev_features

test_features = get_features(test_tokenizer_path, test_filepath)
predict_results = anomaly_detector.predict(test_features)
print(predict_results)
