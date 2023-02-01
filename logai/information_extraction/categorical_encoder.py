#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#

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
from typing import Tuple


@dataclass
class CategoricalEncoderConfig(Config):
    """
    Categorical encoding configurations.
    """

    name: str = "label_encoder"
    params: object = None

    @classmethod
    def from_dict(cls, config_dict):
        config = super(CategoricalEncoderConfig, cls).from_dict(config_dict)
        if config.name.lower() == "one_hot_encoder":
            config.params = OneHotEncodingParams.from_dict(config.params)
        elif config.name.lower() == "ordinal_encoder":
            config.params = OrdinalEncodingParams.from_dict(config.params)
        return config


class CategoricalEncoder:
    """
    Implementation of the categorical encoder.
    """

    def __init__(self, config: CategoricalEncoderConfig):
        """
        Initializes a categorical encoder.

        :param config: Configuration of categorical encoders. Currently support label encoder,
            ordinal encoder, and one-hot encoder.
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

    def fit_transform(self, features: pd.Series) -> Tuple[pd.DataFrame, list]:
        """
        Transforms the str features into categories.

        :param features: A list of features.
        :return: A list of encoded features.
        """
        return self.encoder.fit_transform(features)
