#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import os
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from logai.algorithms.nn_model.transformers import (
    TransformerAlgoConfig,
    TransformerAlgo,
)
from logai.algorithms.parsing_algo.drain import DrainParams
from logai.dataloader.data_loader import DataLoaderConfig, FileDataLoader
from logai.information_extraction.categorical_encoder import (
    CategoricalEncoderConfig,
    CategoricalEncoder,
)
from logai.information_extraction.feature_extractor import (
    FeatureExtractorConfig,
    FeatureExtractor,
)
from logai.information_extraction.log_parser import LogParserConfig, LogParser
from logai.preprocess.preprocess import PreprocessorConfig, Preprocessor
from logai.utils import constants

INPUT_DIR = "/Users/qcheng/workspace/gitsoma/logai/logai/data/open_datasets/HDFS_1/"
OUT_DIR = (
    "/Users/qcheng/workspace/gitsoma/logai/logai/applications/log_bert/test_trainer/"
)

"""

"""
def pre_train_process():
    # dataloader
    filepath = os.path.join(INPUT_DIR, "HDFS.log")
    log_type = "log"
    file_config = DataLoaderConfig(filepath=filepath, log_type=log_type)

    dataloader = FileDataLoader(file_config)
    logrecord = dataloader.load_data()

    print("data loading complete.")

    # preprocess
    loglines = logrecord.body[constants.LOGLINE_NAME]

    preprocessor_config = PreprocessorConfig(
        custom_replace_list=[
            [r"(?<=blk_)[-\d]+", "<block_id>"],
            [r"\d+\.\d+\.\d+\.\d+", "<IP>"],
            [r"(/[-\w]+)+", "<file_path>"],
        ]
    )

    preprocessor = Preprocessor(preprocessor_config)

    clean_logs, custom_patterns = preprocessor.clean_log(loglines)

    print("Preprocess completed")

    # parsing
    parsing_algo_params = DrainParams(sim_th=0.5, depth=5)

    log_parser_config = LogParserConfig(
        parsing_algorithm="drain", parsing_algo_params=parsing_algo_params
    )

    parser = LogParser(log_parser_config)
    parsed_result = parser.parse(clean_logs)

    parsed_result["block_id"] = custom_patterns["<block_id>"].map(
        lambda x: "blk_{}".format(x[0])
    )
    encoder_config = CategoricalEncoderConfig()

    cat_encoder = CategoricalEncoder(encoder_config)

    parsed_result["event_id"] = cat_encoder.fit_transform(
        parsed_result[[constants.PARSED_LOGLINE_NAME]]
    )

    parsed_result.to_pickle(os.path.join(OUT_DIR, "parse_res"))

    parsed_result = pd.read_pickle(os.path.join(OUT_DIR, "parse_res"))

    print("parsing complete.")

    # read labels

    blk_label_file = os.path.join(INPUT_DIR, "anomaly_label.csv")

    blk_df = pd.read_csv(blk_label_file, header=0)
    anomaly_blk = set(blk_df[blk_df["Label"] == "Anomaly"]["BlockId"])

    config = FeatureExtractorConfig(group_by_category=["block_id"])

    feature_extractor = FeatureExtractor(config)

    block_list, seq = feature_extractor.convert_to_sequence(
        parsed_result["event_id"], parsed_result["block_id"]
    )
    block_list["event_sequence"] = seq
    block_list["label"] = block_list["block_id"].apply(
        lambda x: 1 if x in anomaly_blk else 0
    )

    block_list.to_pickle(os.path.join(OUT_DIR, "data_to_train.pkl"))

    print("featurization complete.")

    return


def model_inference():
    block_list = pd.read_pickle(os.path.join(OUT_DIR, "data_to_train.pkl"))
    neg_df = block_list[block_list["label"] == 0][["event_sequence", "label"]].sample(
        frac=0.1, random_state=1
    )
    pos_df = block_list[block_list["label"] == 1][["event_sequence", "label"]].sample(
        frac=0.5, random_state=1
    )

    train_df_neg, test_df_neg = train_test_split(neg_df, test_size=0.2)
    train_df_pos, test_df_pos = train_test_split(pos_df, test_size=0.2)

    test_df = test_df_pos.append(test_df_neg)

    test_logs = test_df["event_sequence"]

    test_logs = test_logs.rename(constants.LOG_EVENTS)

    test_labels = test_df["label"]

    test_df = pd.concat((test_logs, test_labels), axis=1)

    test_dataset = Dataset.from_pandas(test_df)

    model = AutoModelForSequenceClassification.from_pretrained("./test_trainer")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    trainer = Trainer(model=model)

    def tokenize_auto_tokenizer(examples):
        return tokenizer(
            examples[constants.LOG_EVENTS], padding="max_length", truncation=True
        )

    tokenized_test_datasets = test_dataset.map(tokenize_auto_tokenizer, batched=True)

    # TransformerAlgoConfig()
    # transformer = TransformerAlgo(config)

    y = trainer.predict(tokenized_test_datasets)

    print(len(y))
    #
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_train_datasets,
    #     eval_dataset=tokenized_val_datasets,
    #     do_train=False,
    #     do_prediction=True
    #     # compute_metrics=self._compute_metrics,
    # )

    return


def model_train():

    block_list = pd.read_pickle(os.path.join(OUT_DIR, "data_to_train.pkl"))
    neg_df = block_list[block_list["label"] == 0][["event_sequence", "label"]].sample(
        frac=0.1, random_state=1
    )
    pos_df = block_list[block_list["label"] == 1][["event_sequence", "label"]].sample(
        frac=0.5, random_state=1
    )

    train_df_neg, test_df_neg = train_test_split(neg_df, test_size=0.2)
    train_df_pos, test_df_pos = train_test_split(pos_df, test_size=0.2)

    train_df = train_df_pos.append(train_df_neg)
    test_df = test_df_pos.append(test_df_neg)

    config = TransformerAlgoConfig()
    transformer = TransformerAlgo(config)
    transformer.train(train_df["event_sequence"], train_df["label"])

    transformer.save(OUT_DIR)
    return


def main():
    module_to_exec = "predict"
    if module_to_exec == "preprocess":
        pre_train_process()

    elif module_to_exec == "train":
        model_train()

    elif module_to_exec == "predict":
        model_inference()
    else:
        raise ValueError(
            "Invalid module_to_exec: {} Need to specify module_to_exec from: preprocess, train, predict".format(
                module_to_exec
            )
        )
    return


if __name__ == "__main__":
    main()
