#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import torch
import torch.nn.functional as F
from torch import nn
from logai.algorithms.vectorization_algo.forecast_nn import ForecastNNVectorizedDataset

from logai.algorithms.nn_model.forecast_nn.base_nn import (
    ForecastBasedNN,
    ForecastBasedNNParams,
)
from attr import dataclass


@dataclass
class CNNParams(ForecastBasedNNParams):
    """Config for CNN based log representation learning

    Inherits:
        ForecastBasedNNParams: Config class for storing parameters of forecast based neural models

    kernel_sizes: list = [2, 3, 4] # kernel size
    """

    kernel_sizes: list = [2, 3, 4]


class CNN(ForecastBasedNN):
    """CNN based model for learning log representation through a self-supervised forecasting task over log sequences

    Args:
        ForecastBasedNN: base class for forecasting based neural log representation learning
    """

    def __init__(self, config: CNNParams):
        """initialization for CNN based log representation learning

        Args:
            config (CNNParams): parameters for CNN log representation learning model
        """
        super().__init__(config)
        self.config = config
        self.config.model_name = "cnn"
        num_labels = self.meta_data["num_labels"]
        self.hidden_size = self.config.hidden_size

        # if isinstance(self.config.kernel_sizes, str):
        #    self.config.kernel_sizes = list(map(int, self.config.kernel_sizes.split()))

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(1, self.hidden_size, (K, self.config.embedding_dim))
                for K in self.config.kernel_sizes
            ]
        )

        self.criterion = nn.CrossEntropyLoss()
        self.prediction_layer = nn.Linear(
            self.hidden_size * len(self.config.kernel_sizes), num_labels
        )

    def forward(self, input_dict):
        """forward method for cnn model

        Args:
            input_dict (dict): dict containing the session_idx, features, window_anomalies
                and window_labels as in ForecastNNVectorizedDataset object

        Returns:
            dict: dict containing loss and prediction tensor
        """
        if self.label_type == "anomaly":
            y = input_dict[ForecastNNVectorizedDataset.window_anomalies].long().view(-1)
        elif self.label_type == "next_log":
            y = input_dict[ForecastNNVectorizedDataset.window_labels].long().view(-1)
        self.batch_size = y.size()[0]
        x = input_dict[ForecastNNVectorizedDataset.features]
        x = self.embedder(x)

        x = x.unsqueeze(1)

        x = [
            F.relu(conv(x.float())).squeeze(3) for conv in self.convs
        ]  # [(batch_size, hidden_size, seq_len), ...]*len(kernel_sizes)
        x = [
            F.max_pool1d(i, i.size(2)).squeeze(2) for i in x
        ]  # [(batch_size, hidden_size), ...] * len(kernel_sizes)
        representation = torch.cat(x, 1)
        logits = self.prediction_layer(representation)
        y_pred = logits.softmax(dim=-1)
        loss = self.criterion(logits, y)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict
