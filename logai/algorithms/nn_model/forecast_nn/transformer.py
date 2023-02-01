#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from torch import nn
from logai.algorithms.nn_model.forecast_nn.base_nn import (
    ForecastBasedNN,
    ForecastBasedNNParams,
)
from attr import dataclass
from logai.algorithms.vectorization_algo.forecast_nn import ForecastNNVectorizedDataset


@dataclass
class TransformerParams(ForecastBasedNNParams):
    """Config for transformer based log representation learning.

    :param nhead: number of attention heads.
    :param num_layers: number of hidden layers in the neural network.
    """

    nhead: int = 4
    num_layers: int = 1


class Transformer(ForecastBasedNN):
    """Transformer based model for learning log representation through a self-supervised forecasting task over
    log sequences.
    """

    def __init__(self, config: TransformerParams):
        super().__init__(config)
        self.config = config
        self.config.model_name = "transformer"
        num_labels = self.meta_data["num_labels"]
        # self.cls = torch.zeros(1, 1, self.config.embedding_dim).to(self.device)
        encoder_layer = nn.TransformerEncoderLayer(
            self.config.embedding_dim, self.config.nhead, self.config.hidden_size
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.config.num_layers
        )

        self.criterion = nn.CrossEntropyLoss()
        self.prediction_layer = nn.Linear(self.config.embedding_dim, num_labels)

    def forward(self, input_dict):
        """Forward method of transformer based model.

        :param input_dict: dict containing the session_idx, features, window_anomalies and window_labels as in ForecastNNVectorizedDataset object.
        :return: dict containing loss and prediction tensor.
        """
        if self.label_type == "anomaly":
            y = input_dict[ForecastNNVectorizedDataset.window_anomalies].long().view(-1)
        elif self.label_type == "next_log":
            y = input_dict[ForecastNNVectorizedDataset.window_labels].long().view(-1)
        self.batch_size = y.size()[0]
        x = input_dict[ForecastNNVectorizedDataset.features]
        x = self.embedder(x)

        x_t = x.transpose(1, 0)

        x_transformed = self.transformer_encoder(x_t.float())
        representation = x_transformed.transpose(1, 0).mean(dim=1)

        logits = self.prediction_layer(representation)
        y_pred = logits.softmax(dim=-1)
        loss = self.criterion(logits, y)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict
