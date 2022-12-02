#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from logai.algorithms.nn_model.logBERT.unsupervised_log_anomaly_detection.configs import CustomTokenizerConfig
from logai.algorithms.nn_model.logBERT.unsupervised_log_anomaly_detection.train_tokenizer import CustomTokenizer
import os 

def train_tokenizer_pipeline(train_file, custom_tokens, data_column_name, model, output_dir=None):
    if output_dir is None or len(output_dir)==0:
        output_dir = os.path.dirname(train_file)
    
    if model == "bert-base-uncased":
        tokenizer_dirname = "log-bert-uncased-tokenizer"
    elif model == "bert-base-cased":
        tokenizer_dirname = "log-bert-cased-tokenizer"
    tokenizer_dirname = os.path.join(output_dir, tokenizer_dirname)

    tok_config = CustomTokenizerConfig()
    tok_config.tokenizer_config['model_name'] = model 
    tok_config.tokenizer_config['custom_tokens'] = custom_tokens
    custom_tokenizer = CustomTokenizer(tok_config)
    custom_tokenizer.train_tokenizer(train_file, data_column_name, tokenizer_dirname)


if __name__ == "__main__":

    dataset_name = "BGL" # "BGL" #"HDFS"  #"Thunderbird"
    if dataset_name == "HDFS":
        kwargs = {
            'train_file' : "../../datasets/public/HDFS/output/nonparsed/HDFS_train.csv",
            'custom_tokens' : ["[BLOCK]", "[IP]", "[FILE]", "[INT]"],
            'data_column_name' :  "loglines",
            'model' :  "bert-base-cased",
            'output_dir' : ""
        }
    elif dataset_name == "BGL":
        kwargs = {
            'train_file' : "/Users/qcheng/workspace/gitsoma/logai/logai/data/open_datasets/output/nonparsed/BGL_train.csv",
            'custom_tokens' : ["[INT]", "[HEX]", "[IP]"],
            'data_column_name' : "loglines",
            'model' : "bert-base-cased",
            'output_dir' : ""
        }
    elif dataset_name == "Thunderbird":
        kwargs = {
            'train_file' : "../../datasets/public/BGL/output/nonparsed/BGL_train.csv",    
            'custom_tokens' : ["[FILE]", "[HEX]", "[INT]", "[IP]", "[WARNING]", "[ALPHANUM]"],
            'data_column_name' : "loglines",
            'model' : "bert-base-cased",
            'output_dir' : ""
        }


    train_tokenizer_pipeline(**kwargs)
    