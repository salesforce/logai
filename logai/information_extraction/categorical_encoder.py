#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from collections import defaultdict

import pandas as pd
from attr import dataclass

from logai.algorithms.categorical_encoding_algo.label_encoding import LabelEncoding
from logai.algorithms.categorical_encoding_algo.one_hot_encoding import (
    OneHotEncoding,
    OneHotEncodingParams,
)
from logai.algorithms.categorical_encoding_algo.ordinal_encoding import (
    OrdinalEncoding,
    OrdinalEncodingParams,
)
from logai.config_interfaces import Config


@dataclass
class CategoricalEncoderConfig(Config):
    """
    Categorical encoding configurations
    """
    name: str = "label_encoder"
    params: object = None

    def from_dict(self, config_dict):
        super().from_dict(config_dict)

        if self.name.lower() == "one_hot_encoder":
            params = OneHotEncodingParams()
            params.from_dict(self.params)
            self.params = params
        elif self.name.lower() == "ordinal_encoder":
            params = OrdinalEncodingParams()
            params.from_dict(self.params)
            self.params = params


class CategoricalEncoder:
    """
    Implementation of categorical encoder
    """
    def __init__(self, config: CategoricalEncoderConfig):
        """
        Initialize categorical encoder.
        :param config: Configuration of categorical encoders. Currently support:
        - label encoder
        - ordinal encoder
        - one-hot encoder
        """
        self.encoder = None
        if config.name.lower() == "label_encoder":
            self.encoder = LabelEncoding()

        elif config.name.lower() == "one_hot_encoder":
            self.encoder = OneHotEncoding(
                config.params if config.params else OneHotEncodingParams()
            )

        elif config.name.lower() == "ordinal_encoder":
            self.encoder = OrdinalEncoding(
                config.params if config.params else OrdinalEncodingParams()
            )
        else:
            raise RuntimeError(
                "Categorical Encoder {} is not defined".format(config.name)
            )

    def fit_transform(self, features: pd.Series) -> (pd.DataFrame, list):
        """
        Transform the str features into categories.
        :param features: pd.Series: list of features
        :return: pd.Dataframe: list of encoded features.
        """
        return self.encoder.fit_transform(features)
