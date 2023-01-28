#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from schema import Or, Schema, Optional

config_schema = Schema(
    {
        "dataset_name": str,
        Optional("label_filepath"): str,
        "parse_logline": bool,
        "output_dir": str,
        "output_file_type": str,
        "training_type": str,
        "deduplicate_test": bool,
        Optional("test_data_frac_pos"): float,
        Optional("test_data_frac_neg"): float,
        Optional("train_test_shuffle"): bool,
        "data_loader_config": {
            "filepath": str,
            "reader_args": {"log_format": str},
            "log_type": str,
            "dimensions": {
                "body": list,
                "timestamp": list,
                Optional("labels"): Or(list, None),
                Optional("span_id"): list,
            },
            "datetime_format": str,
            "infer_datetime": bool,
        },
        "preprocessor_config": {
            "custom_delimiters_regex": list,
            "custom_replace_list": list,
        },
        "open_set_partitioner_config": {
            "session_window": bool,
            "sliding_window": int,
            "logsequence_delim": str,
        },
        "log_parser_config": {
            "parsing_algorithm": str,
            "parsing_algo_params": {"sim_th": float, "depth": int},
        },
        "log_vectorizer_config": {
            "algo_name": str,
            "algo_param": {
                Optional("model_name"): str,
                Optional("feature_type"): str,
                Optional("sep_token"): str,
                Optional("max_token_len"): int,
                Optional("embedding_dim"): int,
                Optional("label_type"): str,
                Optional("custom_tokens"): list,
                Optional("output_dir"): str,
                Optional("tokenizer_dirname"): str,
                Optional("vectorizer_model_dirpath"): str,
                Optional("vectorizer_metadata_filepath"): str,
            },
        },
        "nn_anomaly_detection_config": {
            "algo_name": str,
            "algo_params": {
                "model_name": str,
                Optional("learning_rate"): float,
                Optional("embedding_dim"): int,
                Optional("max_token_len"): int,
                Optional("feature_type"): str,
                Optional("label_type"): str,
                Optional("eval_type"): str,
                Optional("batch_size"): int,
                Optional("per_device_train_batch_size"): int,
                Optional("num_train_epochs"): int,
                Optional("save_steps"): int,
                Optional("mask_ngram"): int,
                Optional("output_dir"): str,
                Optional("tokenizer_dirpath"): str,
                Optional("metadata_filepath"): str,
            },
        },
    }
)
