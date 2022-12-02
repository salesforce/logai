#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from logai.algorithms.nn_model.logBERT.unsupervised_log_anomaly_detection.predict import LogBERTPredict
from logai.algorithms.nn_model.logBERT.unsupervised_log_anomaly_detection.configs import LogBERTConfig
import pandas as pd 
import os 
import ast 
import torch


def prediction_pipeline(test_file, custom_tokens, max_input_seq_len, lr, model_name, data_column_name, mask_ngram, output_dir):
    
    if output_dir is None or len(output_dir)==0:
        output_dir = os.path.dirname(test_file)

    model_name = model_name+'_finetuned_custom-tokenizer_lr'+str(lr)+'_maxlen'+str(max_input_seq_len)
    model_dir = os.path.join(output_dir, 'models', model_name)
    
    if "bert-base-uncased" in model_name:
        tokenizer_dirname = "log-bert-uncased-tokenizer"
    elif "bert-base-cased" in model_name:
        tokenizer_dirname = "log-bert-cased-tokenizer"
    tokenizer_dirname = os.path.abspath(os.path.join(output_dir, tokenizer_dirname))

    output_dir = os.path.join(output_dir, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = os.path.join(output_dir, model_name+'.csv')

    #if not os.path.exists(output_filename):
    logbert_config = LogBERTConfig()
    logbert_config.tokenizer_config['tokenizer_name_or_dir'] = tokenizer_dirname
    logbert_config.trainer_config['model_dir'] = model_dir
    logbert_config.tokenizer_config['max_input_seq_len'] = max_input_seq_len
    logbert_config.tokenizer_config['custom_tokens'] = custom_tokens
    logbert_config.data_config['data_column_name'] = data_column_name
    logbert_config.eval_config['mask_ngram'] = mask_ngram

    logbert_eval = LogBERTPredict(logbert_config)
    logbert_eval.predict(test_file, output_filename)

    '''    
    elif os.path.exists(output_filename):
        eval_metrics_per_instance_series = pd.read_csv(output_filename)
        eval_metrics_per_instance_series['top6_loss'] = eval_metrics_per_instance_series['top6_loss'].apply(ast.literal_eval)
        eval_metrics_per_instance_series['top6_max_prob'] = eval_metrics_per_instance_series['top6_max_prob'].apply(ast.literal_eval)
        eval_metrics_per_instance_series['top6_min_logprob'] = eval_metrics_per_instance_series['top6_min_logprob'].apply(ast.literal_eval)
        eval_metrics_per_instance_series['top6_max_entropy'] = eval_metrics_per_instance_series['top6_max_entropy'].apply(ast.literal_eval)
        
        compute_metrics(eval_metrics_per_instance_series, test_labels)
    '''



if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    dataset_name = "BGL"  #"HDFS" #BGL #"Thunderbird"
    
    if dataset_name == "HDFS":
        kwargs = {
            'test_file' : "../../datasets/public/HDFS/output/nonparsed/HDFS_test.csv",
            'custom_tokens' : ["[BLOCK]", "[IP]", "[FILE]", "[INT]"],
            'max_input_seq_len' : 384,
            'lr' : 1e-5,
            'model_name' : "bert-base-cased",
            'data_column_name' : 'loglines',
            'mask_ngram' : 8,
            'output_dir' : ''
        }
    elif dataset_name == "BGL":
        kwargs = {
            'test_file' : "/Users/qcheng/workspace/gitsoma/logai/logai/data/open_datasets/output/nonparsed/BGL_test.csv",
            'custom_tokens' : ["[INT]", "[HEX]", "[IP]"],
            'max_input_seq_len' : 120,
            'lr' : 1e-5,
            'model_name' : "bert-base-cased",
            'data_column_name' : 'loglines',
            'mask_ngram' : 1,
            'output_dir' : ''
        }
    elif dataset_name == "Thunderbird":
        kwargs = {
            'test_file' : "../../datasets/public/Thunderbird/output/nonparsed/Thunderbird_test.csv",
            'max_input_seq_len' : 120,
            'custom_tokens': ["[FILE]", "[HEX]", "[INT]", "[IP]", "[WARNING]", "[ALPHANUM]"],
            'lr' : 1e-4,
            'model_name' : "bert-base-cased",
            'data_column_name' : 'loglines',
            'mask_ngram' : 1,
            'output_dir' : ''
        }

    prediction_pipeline(**kwargs)

