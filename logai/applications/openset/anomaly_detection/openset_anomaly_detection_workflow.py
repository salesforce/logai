#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from logai.utils import constants
from attr import dataclass
from logai.dataloader.data_loader import FileDataLoader
from logai.dataloader.data_model import LogRecordObject
from logai.utils.dataset_utils import split_train_dev_test_for_anomaly_detection
from logai.information_extraction.log_parser import LogParser
from logai.preprocess.openset_partitioner import OpenSetPartitioner
from logai.applications.application_interfaces import WorkFlowConfig
from logai.utils.file_utils import *
from logai.preprocess.hdfs_preprocessor import HDFSPreprocessor
from logai.preprocess.bgl_preprocessor import BGLPreprocessor
from logai.preprocess.thunderbird_preprocessor import ThunderbirdPreprocessor
from logai.analysis.nn_anomaly_detector import NNAnomalyDetector
from logai.information_extraction.log_vectorizer import LogVectorizer
from .configs.schema import config_schema
import logging
from schema import SchemaError

from logai.information_extraction.feature_extractor import (
    FeatureExtractor,
    FeatureExtractorConfig,
)

logging.basicConfig(level=logging.INFO)


def validate_config_dict(workflow_config_dict):
    """Method to validate the config dict with the schema

    :param workflow_config_dict: (dict): dict containing config for anomaly detection workflow on open log datasets
    """
    try:
        config_schema.validate(workflow_config_dict)
    except SchemaError as se:
        raise se


def get_openset_ad_config(
    config_filename: str,
    anomaly_detection_type: str,
    vectorizer_type: str,
    parse_logline: bool,
    training_type: str,
):
    """Method to dynamically set some of the config parameters based on the given arguments. List of all possible supported anomaly detection types and vectorizer types configurations can be found in the config yaml file. Avoid this function if you are directly setting all config parameters manually
    
    :param config_filename: (str): Name of the config file (currently supports hdfs and bgl)
    :param anomaly_detection_type: (str): string describing the type of anomaly detection
    :param vectorizer_type: (str): string describing the type of vectorizer.
    :param parse_logline: (bool): Whether to use log parsing or not
    :param training_type: (str): Whether to use "supervised" or "unsupervised" training
    :return: OpenSetADWorkflowConfig: config object of type OpenSetADWorkflowConfig
    """
    config_path = os.path.join(
        os.path.dirname(__file__), "configs", "{}.yaml".format(config_filename)
    )

    config_parsed = read_file(config_path)
    workflow_config_dict = config_parsed["workflow_config"]

    if parse_logline is not None:
        workflow_config_dict["parse_logline"] = parse_logline

    if workflow_config_dict["training_type"] is None:
        workflow_config_dict["training_type"] = training_type

    workflow_config_dict["nn_anomaly_detection_config"] = workflow_config_dict[
        "nn_anomaly_detection_config"
    ][anomaly_detection_type]
    workflow_config_dict["log_vectorizer_config"] = workflow_config_dict[
        "log_vectorizer_config"
    ][vectorizer_type]

    if "forecast_nn" in vectorizer_type:
        if workflow_config_dict["training_type"] == "supervised":
            workflow_config_dict["log_vectorizer_config"]["algo_param"][
                "label_type"
            ] = "anomaly"
        elif workflow_config_dict["training_type"] == "unsupervised":
            workflow_config_dict["log_vectorizer_config"]["algo_param"][
                "label_type"
            ] = "next_log"
        workflow_config_dict["open_set_partitioner_config"] = workflow_config_dict[
            "open_set_partitioner_config"
        ]["forecast_nn"]
    elif "logbert" in vectorizer_type:
        workflow_config_dict["open_set_partitioner_config"] = workflow_config_dict[
            "open_set_partitioner_config"
        ]["logbert"]

    # validate_config_dict(workflow_config_dict)
    workflow_config = OpenSetADWorkflowConfig.from_dict(workflow_config_dict)
    return workflow_config


@dataclass
class OpenSetADWorkflowConfig(WorkFlowConfig):
    """Config for Log Anomaly Detection workflow on Open Log dataset Inherits: WorkFlowConfig: Config object for specifying workflow parameters

    :param dataset_name: str = None: name of the public open dataset
    :param label_filepath: str = None: path to the separate file (if any) containing the anomaly detection labels
    :param output_dir: str = None : path to output directory where all intermediate and final outputs would be dumped
    :param parse_logline: bool = False : whether to parse or not
    :param training_type: str = None: should be either supervised or unsupervised
    :param deduplicate_test: bool = False : whether to de-duplicate the instances in the test data, while maintaining a count of the number of each duplicated instance
    :param test_data_frac_pos: float = 0.8 : fraction of the logs having positive class used for teest
    :param test_data_frac_neg: float = 0.8 : fraction of the logs having negative class used for test
    :param train_test_shuffle: bool = False : whether to use chronological ordering of the logs or to shuffle them when creating the train test splits
    """

    dataset_name: str = None  # name of the public open dataset
    label_filepath: str = None  # path to the separate file (if any) containing the anomaly detection labels
    output_dir: str = None  # path to output directory where all intermediate and final outputs would be dumped
    parse_logline: bool = False  # whether to parse or not
    training_type: str = None  # should be either supervised or unsupervised
    deduplicate_test: bool = False  # whether to de-duplicate the instances in the test data, while maintaining
                                    # a count of the number of each duplicated instance
    test_data_frac_pos: float = (
        0.8  # fraction of the logs having positive class used for teest
    )
    test_data_frac_neg: float = (
        0.8  # fraction of the logs having negative class used for test
    )
    train_test_shuffle: bool = False  # whether to use chronological ordering of the logs or
                                      # to shuffle them when creating the train test splits


class OpenSetADWorkflow:
    """log anomaly detection workflow for open log datasets

    :param config: (OpenSetADWorkflowConfig): config object specifying parameters for log anomaly detection over open datasets
    """
    def __init__(self, config: OpenSetADWorkflowConfig):
        
        self.config = config

    def _get_parse_type_str(self):
        if self.config.parse_logline:
            parse_type = "parsed"
        else:
            parse_type = "nonparsed"
        return parse_type

    def _get_partition_type_str(self):
        if (
            self.config.open_set_partitioner_config.sliding_window == 0
            and self.config.open_set_partitioner_config.session_window
        ):
            partition_type = "session"
        else:
            partition_type = "sliding" + str(
                self.config.open_set_partitioner_config.sliding_window
            )
        return partition_type

    def _get_training_type_str(self):
        return self.config.training_type

    def _get_output_filename(self, suffix):
        f = pathlib.Path(self.config.data_loader_config.filepath)
        basename = f.stem + "_" + suffix
        ext = ".csv"
        filepath = basename + ext
        return os.path.join(self.config.output_dir, filepath)

    def load_dataloader(self):
        """initialize dataloader object"""
        self.dataloader = FileDataLoader(self.config.data_loader_config)

    def load_preprocessor(self):
        """initialize preprocessor object

        Raises:
            ValueError: dataset is not supported
        """
        if self.config.dataset_name == "hdfs":
            self.preprocessor = HDFSPreprocessor(
                self.config.preprocessor_config, self.config.label_filepath
            )
        elif self.config.dataset_name == "bgl":
            self.preprocessor = BGLPreprocessor(self.config.preprocessor_config)
        elif self.config.dataset_name == "thunderbird":
            self.preprocessor = ThunderbirdPreprocessor(self.config.preprocessor_config)
        else:
            raise ValueError(
                "dataset name {} not supported ".format(self.config.dataset_name)
            )

    def load_parser(self):
        """initialize log parser object"""
        self.parser = LogParser(self.config.log_parser_config)

    def load_partitioner(self):
        """initialize partitioner object"""
        self.partitioner = OpenSetPartitioner(self.config.open_set_partitioner_config)

    def load_deduper(self):
        """initialize dedup object"""
        fe_config = FeatureExtractorConfig.from_dict(
            {"group_by_category": [constants.SPAN_ID, constants.LOGLINE_NAME]}
        )
        self.feature_extractor = FeatureExtractor(fe_config)

    def load_vectorizer(self):
        """initialize vectorizer object"""
        if self.config.log_vectorizer_config.algo_param.output_dir == "":
            self.config.log_vectorizer_config.algo_param.output_dir = os.path.join(
                self.config.output_dir,
                self._get_parse_type_str()
                + "_"
                + self._get_partition_type_str()
                + "_"
                + self._get_training_type_str()
                + "_AD",
            )
        self.vectorized_data_dirpath = os.path.join(
            self.config.log_vectorizer_config.algo_param.output_dir, "vectorized_data"
        )

        if not os.path.exists(self.vectorized_data_dirpath):
            os.makedirs(self.vectorized_data_dirpath)

        self.vectorizer = LogVectorizer(self.config.log_vectorizer_config)

    def load_anomaly_detector(self):
        """initialize anomaly detector object"""
        self.set_anomaly_detector_configs()
        self.anomaly_detector = NNAnomalyDetector(
            config=self.config.nn_anomaly_detection_config
        )

    def load_data(self):
        """loads logrecord object from raw log dataset

        :return: LogRecordObject : logrecord object created from the raw log dataset
        """
        self.load_dataloader()
        logrecord = self.dataloader.load_data()
        logging.info(
            "Loaded log record object from {} ".format(
                self.config.data_loader_config.filepath
            )
        )
        return logrecord

    def preprocess_log_data(self, logrecord):
        """preprocesses logrecord object by doing custom dataset specific data cleaning and formatting

        :param logrecord: (LogRecordObject): log record object to be preprocessed
        :return:  LogRecordObject: preprocessed lgo record object using custom dataset-specific preprocessing
        """
        self.load_preprocessor()
        preprocessed_filepath = self._get_output_filename(suffix="preprocessed")
        if not file_exists(preprocessed_filepath):
            logrecord = self.preprocessor.clean_log(logrecord)
            logrecord.save_to_csv(preprocessed_filepath)
            logging.info(
                "Finished preprocessing ... saved preprocessed data in {}".format(
                    preprocessed_filepath
                )
            )
        else:
            logrecord = LogRecordObject.load_from_csv(preprocessed_filepath)
            logging.info(
                "Loaded preprocessed data from {}".format(preprocessed_filepath)
            )
        return logrecord

    def parse_log_data(self, logrecord):
        """parse logrecord object by applying standard log parsers as specified in the Config

        :param logrecord: (LogRecordObject): logrecord object to be parsed
        :return: LogRecordObject: parsed logrecord object
        """
        self.load_parser()
        parsed_filepath = self._get_output_filename(suffix=self._get_parse_type_str())
        if not file_exists(parsed_filepath):
            parsed_result = self.parser.parse(logrecord.body[constants.LOGLINE_NAME])
            logrecord.body[constants.LOGLINE_NAME] = parsed_result[
                constants.PARSED_LOGLINE_NAME
            ]
            logrecord.save_to_csv(parsed_filepath)

            logging.info(
                "Finished parsing.. saved parsed data in {}".format(parsed_filepath)
            )
        else:
            logrecord = LogRecordObject.load_from_csv(parsed_filepath)
            logging.info("Loaded past parsed data from {}".format(parsed_filepath))
        return logrecord

    def partition_log_data(self, logrecord: LogRecordObject):
        """partitioning logrecord object by applying session or sliding window based partitions

        :param logrecord: (LogRecordObject): logrecord object to be partitioned
        :return: logrecord: partitioned logrecord object
        """
        self.load_partitioner()
        output_filepath_suffix = (
            self._get_parse_type_str() + "_" + self._get_partition_type_str()
        )
        partitioned_filepath = self._get_output_filename(suffix=output_filepath_suffix)
        if not file_exists(partitioned_filepath):
            logrecord = self.partitioner.partition(logrecord)
            logrecord.save_to_csv(partitioned_filepath)
            logging.info(
                "Finished partitioning.. saved partitioned data in {}".format(
                    partitioned_filepath
                )
            )
        else:
            logrecord = LogRecordObject.load_from_csv(partitioned_filepath)
            logging.info("Loaded partitioned data from {}".format(partitioned_filepath))
        return logrecord

    def generate_train_dev_test_data(self, logrecord: LogRecordObject):
        """splitting open log datasets into train dev and test splits according to the parameters specified in the config object

        :param logrecord: (LogRecordObject): logrecord object to be split into train, dev and test
        :return: - train_data: logrecord object containing training dataset.
            - dev_data: logrecord object containing dev dataset.
            - test_data: logrecord object containing test dataset
        """
        output_filepath_suffix = (
            self._get_parse_type_str()
            + "_"
            + self._get_partition_type_str()
            + "_"
            + self._get_training_type_str()
        )
        train_filepath = self._get_output_filename(
            suffix=output_filepath_suffix + "_train"
        )
        dev_filepath = self._get_output_filename(suffix=output_filepath_suffix + "_dev")
        test_filepath = self._get_output_filename(
            suffix=output_filepath_suffix + "_test"
        )
        if not (
            file_exists(train_filepath)
            and file_exists(dev_filepath)
            and file_exists(test_filepath)
        ):
            (
                train_data,
                dev_data,
                test_data,
            ) = split_train_dev_test_for_anomaly_detection(
                logrecord,
                training_type=self.config.training_type,
                test_data_frac_neg_class=self.config.test_data_frac_neg,
                test_data_frac_pos_class=self.config.test_data_frac_pos,
                shuffle=self.config.train_test_shuffle,
            )
            train_data.save_to_csv(train_filepath)
            dev_data.save_to_csv(dev_filepath)
            test_data.save_to_csv(test_filepath)
            logging.info("Created train data .. saved in {}".format(train_filepath))
            logging.info("Created dev data .. saved in {}".format(dev_filepath))
            logging.info("Created test data .. saved in {}".format(test_filepath))
        else:
            train_data = LogRecordObject.load_from_csv(train_filepath)
            dev_data = LogRecordObject.load_from_csv(dev_filepath)
            test_data = LogRecordObject.load_from_csv(test_filepath)
            logging.info("Loaded past train data from {}".format(train_filepath))
            logging.info("Loaded past dev data from {}".format(dev_filepath))
            logging.info("Loaded past test data from {}".format(test_filepath))

        return train_data, dev_data, test_data

    def dedup_data(self, logrecord: LogRecordObject):
        """Method to run deduplication of log records, where loglines having same body and span id is collapsed into a single logline. The original occurrent count values of theseloglines is added as a pandas Series object in the 'attributes' property of the logrecord object.

        :param logrecord: (LogRecordObject): logrecord object to be deduplicated
        :return: LogRecordObject: resulting logrecord object
        """
        self.load_deduper()
        old_data_len = len(logrecord.body)
        df_new = self.feature_extractor.convert_to_counter_vector(
            log_pattern=logrecord.body[constants.LOGLINE_NAME],
            attributes=logrecord.labels.join(logrecord.span_id),
            timestamps=logrecord.timestamp[constants.LOG_TIMESTAMPS],
        )
        df_new[constants.LABELS] = df_new[constants.LOG_TIMESTAMPS].apply(
            lambda x: int(sum(x) > 0)
        )
        df_new[constants.LOG_TIMESTAMPS] = df_new[constants.LOG_TIMESTAMPS].apply(
            lambda x: x[-1]
        )
        meta_data = {
            constants.Field.BODY: [constants.LOGLINE_NAME],
            constants.Field.LABELS: [constants.LABELS],
            constants.Field.SPAN_ID: [constants.SPAN_ID],
            constants.Field.ATTRIBUTES: [constants.LOG_COUNTS],
            constants.Field.TIMESTAMP: [constants.LOG_TIMESTAMPS],
        }
        new_data_len = len(df_new)
        logrecord_new = LogRecordObject().from_dataframe(df_new, meta_data=meta_data)
        logging.info(
            "Reduced data from {} to {} by removing duplicates".format(
                old_data_len, new_data_len
            )
        )
        return logrecord_new

    def run_data_processing_workflow(self):
        """Running data processing pipeline for log anomaly detection workflow

        :return: - train_data: logrecord object containing training dataset.
            - dev_data: logrecord object containing dev dataset.
            - test_data: logrecord object containing test dataset
        """
        logrecord = self.load_data()
        logrecord = self.preprocess_log_data(logrecord=logrecord)
        if self.config.parse_logline:
            logrecord = self.parse_log_data(logrecord=logrecord)
        logrecord = self.partition_log_data(logrecord=logrecord)
        train_data, dev_data, test_data = self.generate_train_dev_test_data(logrecord)
        train_data = train_data.dropna()
        dev_data = dev_data.dropna()
        test_data = test_data.dropna()
        if (
            self.config.deduplicate_test
            and "logbert" in self.config.log_vectorizer_config.algo_name
        ):
            test_data = self.dedup_data(test_data)
        return train_data, dev_data, test_data

    def vectorizer_transform(self, logrecord: LogRecordObject, output_filename=None):
        """Applying vectorization on a logrecord object based on the kind of vectorizer specific in Config

        :param logrecord: (LogRecordObject): logrecord containing data to be vectorized
        :param output_filename: (str, optional): path to output file where the vectorized log data would be dumped. Defaults to None.
        :return: vectorized_output : vectorized data
        """
        if output_filename and os.path.exists(output_filename):
            vectorized_output = pkl.load(open(output_filename, "rb"))
        else:
            vectorized_output = self.vectorizer.transform(logrecord)
            if output_filename:
                pkl.dump(vectorized_output, open(output_filename, "wb"))
        return vectorized_output

    def run_vectorizer(self, train_logrecord, dev_logrecord, test_logrecord):
        """Wrapper method for applying vectorization on train, dev and test logrecord objects

        :param train_logrecord: (LogRecordObject): logrecord object of the training dataset
        :param dev_logrecord: (LogRecordObject): logrecord object of the dev dataset
        :param test_logrecord: (LogRecordObject): logrecord object of the test dataset
        :return: - train_data : vectorized train data.
            - dev_data: vectorized dev data.
            - test_data: vectorized test data.
        """
        self.load_vectorizer()
        self.vectorizer.fit(train_logrecord)
        vectorized_train_filename = None
        vectorized_dev_filename = None
        vectorized_test_filename = None
        if self.config.log_vectorizer_config.algo_name == "forecast_nn":
            vectorized_train_filename = os.path.join(
                self.vectorized_data_dirpath, "train.pkl"
            )
            vectorized_dev_filename = os.path.join(
                self.vectorized_data_dirpath, "dev.pkl"
            )
            vectorized_test_filename = os.path.join(
                self.vectorized_data_dirpath, "test.pkl"
            )
        train_data = self.vectorizer_transform(
            train_logrecord, output_filename=vectorized_train_filename
        )
        logging.info("Converted train data to vectors")
        dev_data = self.vectorizer_transform(
            dev_logrecord, output_filename=vectorized_dev_filename
        )
        logging.info("Converted dev data to vectors")
        test_data = self.vectorizer_transform(
            test_logrecord, output_filename=vectorized_test_filename
        )
        logging.info("Converted test data to vectors")

        return train_data, dev_data, test_data

    def run_anomaly_detection(self, train_data, dev_data, test_data):
        """Method to train and run inference of anomaly detector

        :param train_data: vectorized version of the train dataset
        :param dev_data: vectorized version of the dev dataset
        :param test_data: vectorized version of the test dataset
        """
        self.load_anomaly_detector()
        self.anomaly_detector.fit(train_data, dev_data)
        logging.info("Trained anomaly detector")
        predict_results = self.anomaly_detector.predict(test_data)
        logging.info("Ran inference on anomaly detector")

    def set_anomaly_detector_configs(self):
        """setting anomaly detector model configs based on the vectorizer configs"""
        vectorizer_params = self.config.log_vectorizer_config.algo_param
        model_params = self.config.nn_anomaly_detection_config.algo_params

        if self.config.log_vectorizer_config.algo_name == "forecast_nn":
            model_params.embedding_dim = vectorizer_params.embedding_dim
            model_params.feature_type = vectorizer_params.feature_type
            model_params.label_type = vectorizer_params.label_type
            model_params.output_dir = vectorizer_params.output_dir
            model_params.metadata_filepath = (
                vectorizer_params.vectorizer_metadata_filepath
            )

            if self.config.nn_anomaly_detection_config.algo_name == "lstm":
                model_params.max_token_len = vectorizer_params.max_token_len

        elif self.config.log_vectorizer_config.algo_name == "logbert":
            model_params.tokenizer_dirpath = vectorizer_params.tokenizer_dirpath
            model_params.model_name = vectorizer_params.model_name
            model_params.output_dir = vectorizer_params.output_dir

    def execute(self):
        """Method to execute the end to end workflow for anomaly detection on open log datasets"""
        logging.info("Going to data processing")
        (
            train_logrecord,
            dev_logrecord,
            test_logrecord,
        ) = self.run_data_processing_workflow()
        logging.info("Going to vectorize logs")
        train_data, dev_data, test_data = self.run_vectorizer(
            train_logrecord, dev_logrecord, test_logrecord
        )
        logging.info("Going to Anomaly Detection")
        self.run_anomaly_detection(train_data, dev_data, test_data)
