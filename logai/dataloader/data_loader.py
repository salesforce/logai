#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import logging
import re

import pandas as pd
from attr import dataclass

from logai.config_interfaces import Config
from logai.dataloader.data_model import LogRecordObject
from logai.utils import constants


@dataclass
class DataLoaderConfig(Config):
    """
    The configuration class of data loader.
    """

    filepath: str = ""
    log_type: str = "csv"
    dimensions: dict = dict()
    reader_args: dict = dict()
    infer_datetime: bool = False
    datetime_format: str = "%Y-%M-%dT%H:%M:%SZ"  # Default the ISO 8601 format example 2022-05-26T21:29:09+00:00
    open_dataset: str = None


class FileDataLoader:
    """
    Implementation of file data loader, reading log record objects from local files.
    """

    def __init__(self, config: DataLoaderConfig):
        """
        Initializes FileDataLoader by consuming the configuration.
        """
        self.config = config

    def load_data(self) -> LogRecordObject:
        """
        Loads log data with given configuration.
        Currently support file formats:
        - csv
        - tsv
        - other plain text format such as .log with proper parsing configurations

        :return: The logs read from log files and converted into LogRecordObject.
        """
        kwargs = self.config.reader_args
        fpath = self.config.filepath
        if self.config.log_type == "csv":
            df = pd.read_csv(fpath, **kwargs)
        elif self.config.log_type == "tsv":
            df = pd.read_table(fpath, **kwargs)
        elif self.config.log_type == "json":
            df = pd.read_json(fpath, **kwargs)
        else:
            df = self._read_logs(fpath)

        return self._create_log_record_object(df)

    def _read_logs(self, fpath):
        if "log_format" not in self.config.reader_args.keys():
            raise RuntimeError("log_format is needed to read free-form-text logs.")
        log_format = self.config.reader_args["log_format"]

        df = self._log_to_dataframe(fpath, log_format)

        return df

    def _log_to_dataframe(self, fpath, log_format):
        """
        Function to transform log file to dataframe.
        """
        headers = []
        splitters = re.split(r"(<[^<>]+>)", log_format)
        regex = ""
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(" +", "\\\s+", splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip("<").strip(">")
                regex += "(?P<%s>.*?)" % header
                headers.append(header)
        regex = re.compile("^" + regex + "$")

        log_messages = []
        cnt = 0
        with open(fpath, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                cnt += 1
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                except Exception as e:
                    logging.error("Read log file failed. Exception {}.".format(e))

        logdf = pd.DataFrame(log_messages, columns=headers, dtype=str)
        return logdf

    def _create_log_record_object(self, df: pd.DataFrame):
        dims = self.config.dimensions
        log_record = LogRecordObject()
        # Read all available log fields from config.

        if not dims:
            selected = pd.DataFrame(
                df.agg(lambda x: " ".join(x.values), axis=1).rename(
                    constants.LOGLINE_NAME
                )
            )
            setattr(log_record, "body", selected)
        else:
            for field in LogRecordObject.__dataclass_fields__:
                if field in dims.keys():
                    selected = df[list(dims[field])]
                    if field == "body":
                        if len(selected.columns) > 1:
                            selected = pd.DataFrame(
                                selected.agg(
                                    lambda x: " ".join(x.values), axis=1
                                ).rename(constants.LOGLINE_NAME)
                            )
                        else:
                            selected.columns = [constants.LOGLINE_NAME]
                    if field == "span_id":
                        if len(selected.columns) > 1:
                            raise RuntimeError("span_id should be single column")
                        selected.columns = [constants.SPAN_ID]

                    if field == "labels":
                        selected.columns = [constants.Field.LABELS]

                    if field == "timestamp":
                        if len(selected.columns) > 1:
                            selected = pd.DataFrame(
                                selected.agg(
                                    lambda x: " ".join(x.values), axis=1
                                ).rename(constants.LOG_TIMESTAMPS)
                            )
                        selected.columns = [constants.LOG_TIMESTAMPS]
                        if self.config.infer_datetime and self.config.datetime_format:
                            datetime_format = self.config.datetime_format
                            selected[constants.LOG_TIMESTAMPS] = pd.to_datetime(
                                selected[constants.LOG_TIMESTAMPS],
                                format=datetime_format,
                            )

                    setattr(log_record, field, selected)
        # log_record.__post_init__()
        return log_record


class DefaultDataLoader:
    # TODO: Placeholder to implement connector based data loader.
    def __init__(self):
        """
        Initializes default data loader.
        """
        self._logger = logging.Logger()
