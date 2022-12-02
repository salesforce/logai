#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from logai.algorithms.nn_model.logBERT.unsupervised_log_anomaly_detection.configs import LogBERTConfig
from logai.algorithms.nn_model.logBERT.unsupervised_log_anomaly_detection.train import LogBERT
import os 

def training_pipeline(train_file, valid_file, max_input_seq_len, custom_tokens, lr, model_name, data_column_name, output_dir=None):
    if output_dir is None or len(output_dir)==0:
        output_dir = os.path.dirname(train_file)
    if model_name ==  "bert-base-uncased":
        tokenizer_dirname = "log-bert-uncased-tokenizer"
    elif model_name == "bert-base-cased":
        tokenizer_dirname = "log-bert-cased-tokenizer"
    tokenizer_dirname = os.path.abspath(os.path.join(output_dir, tokenizer_dirname))

    print ('Taking custom_tokenizer from ',tokenizer_dirname)
    model_dir = os.path.join(output_dir, 'models', model_name+'_finetuned_custom-tokenizer_lr'+str(lr)+'_maxlen'+str(max_input_seq_len))

    logbert_config = LogBERTConfig()
    logbert_config.trainer_config['model_name'] = model_name
    logbert_config.tokenizer_config['tokenizer_name_or_dir'] = tokenizer_dirname
    logbert_config.tokenizer_config['max_input_seq_len'] = max_input_seq_len
    logbert_config.tokenizer_config['custom_tokens'] = custom_tokens
    logbert_config.trainer_config['learning_rate'] = lr
    logbert_config.trainer_config['model_dir'] = model_dir
    logbert_config.data_config['data_column_name'] = data_column_name

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    logbert = LogBERT(logbert_config)
    logbert.train(train_file, valid_file)


if __name__ == "__main__":
    #HDFS
    dataset_name = "BGL" #BGL #HDFS  #"Thunderbird"

    if dataset_name == "HDFS":
        kwargs = {
            'train_file': "../../datasets/public/HDFS/output/nonparsed/HDFS_train.csv",
            'valid_file': "../../datasets/public/HDFS/output/nonparsed/HDFS_dev.csv",
            'max_input_seq_len': 384,
            'custom_tokens' : ["[INT]", "[IP]", "[BLOCK]", "[FILE]"],
            'lr': 1e-5,
            'model_name': "bert-base-cased",
            'data_column_name': 'loglines',
            'output_dir': ''
        }
    elif dataset_name == "BGL":
        kwargs = {
            'train_file' : "/Users/qcheng/workspace/gitsoma/logai/logai/data/open_datasets/output/nonparsed/BGL_train.csv",
            'valid_file' : "/Users/qcheng/workspace/gitsoma/logai/logai/data/open_datasets/output/nonparsed/BGL_dev.csv",
            'max_input_seq_len' : 120,
            'custom_tokens': ["[HEX]", "[INT]", "[IP]"],
            'lr' : 1e-5,
            'model_name' : "bert-base-cased",
            'data_column_name' : 'loglines',
            'output_dir' : ''
        }
    elif dataset_name == "Thunderbird":
        kwargs = {
            'train_file' : "../../datasets/public/BGL/output/nonparsed/BGL_train.csv",
            'valid_file' : "../../datasets/public/BGL/output/nonparsed/BGL_dev.csv",
            'max_input_seq_len' : 120,
            'custom_tokens': ["[FILE]", "[HEX]", "[INT]", "[IP]", "[WARNING]", "[ALPHANUM]"],
            'lr' : 1e-4,
            'model_name' : "bert-base-cased",
            'data_column_name' : 'loglines',
            'output_dir' : ''
        }

    
    training_pipeline(**kwargs)