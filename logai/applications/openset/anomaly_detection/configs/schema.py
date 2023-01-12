from schema import Or, Schema, Optional

config_schema = Schema(
    {
        "dataset_name": str,
        "label_filepath": str,
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
            },
            "datetime_format": str,
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
                "feature_type": str,
                "sep_token": str,
                "max_token_len": int,
                "embedding_dim": int,
                "label_type": str,
                Optional("output_dir"): str,
                Optional("tokenizer_dirpath"): str,
            },
        },
        "anomaly_detection_config": {
            "algo_name": str,
            "algo_params": {
                "model_name": str,
                "learning_rate": float,
                Optional("embedding_dim"): int,
                Optional("feature_type"): str,
                Optional("label_type"): str,
                Optional("output_dir"): str,
                Optional("model_name"): str,
                Optional("tokenizer_dirpath"): str,
            },
        },
    }
)
