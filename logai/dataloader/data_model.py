#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class LogRecordObject:
    """
    Log record object data model.
    """
    timestamp: pd.DataFrame = pd.DataFrame()
    attributes: pd.DataFrame = pd.DataFrame()
    resource: pd.DataFrame = pd.DataFrame()
    trace_id: pd.DataFrame = pd.DataFrame()
    span_id: pd.DataFrame = pd.DataFrame()
    severity_text: pd.DataFrame = pd.DataFrame()
    severity_number: pd.DataFrame = pd.DataFrame()
    body: pd.DataFrame = pd.DataFrame()
    labels: pd.DataFrame = pd.DataFrame()
    _index: np.array = field(init=False)

    def __post_init__(self):
        self._index = pd.DataFrame(self.body.index.values)

        for field in self.__dataclass_fields__:
            field_content = getattr(self, field)
            if not field_content.empty:
                if not field_content.index.equals(self._index.index):
                    raise IndexError("Index of {} should match Index of this object".format(field))

    def to_dataframe(self):
        """
        Generate pandas.DataFrame from LogRecordType
        :return:
        """
        if self.body.empty:
            return None
        df = self.body
        for field in self.__dataclass_fields__:
            if field == "body":
                continue
            field_content = getattr(self, field)
            if not field_content.empty:
                df = df.join(field_content)

        return df

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame, meta_data: dict = None):
        """
        Convert pandas.DataFrame to log record object
        :param data: pd.DataFrame: log data in pandas dataframe
        :param meta_data: dict: a dictionary that maps data.columns to fields of LogRecordObject
        :return: LogRecordObject
        """

        if meta_data is None and not data.empty:
            logbody = data.apply(" ".join, axis=1)
            return LogRecordObject(body=logbody)

        logrecord = LogRecordObject()

        for key in meta_data:
            if key not in logrecord.__dataclass_field__.keys():
                raise KeyError("{} is not a field in LogRecordObject. The valid fields: {}".format(
                    key, logrecord.__dataclass_field__.keys())
                )
            logrecord.__setattr__(key, data[key])

        return logrecord
