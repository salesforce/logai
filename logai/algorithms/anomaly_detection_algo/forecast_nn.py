#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from logai.algorithms.algo_interfaces import NNAnomalyDetectionAlgo
from logai.algorithms.nn_model.forecast_nn.base_nn import ForecastBasedNNParams
from logai.algorithms.vectorization_algo.forecast_nn import ForecastNNVectorizedDataset
from logai.algorithms.nn_model.forecast_nn.lstm import LSTM, LSTMParams
from logai.algorithms.nn_model.forecast_nn.cnn import CNN, CNNParams
from logai.algorithms.nn_model.forecast_nn.transformer import (
    Transformer,
    TransformerParams,
)
from logai.algorithms.factory import factory
from torch.utils.data import DataLoader


class ForcastBasedNeuralAD(NNAnomalyDetectionAlgo):
    """Forcasting based neural anomaly detection models taken from the deep-loglizer paper
    (https://arxiv.org/pdf/2107.05908.pdf)
    """

    def __init__(self, config: ForecastBasedNNParams):
        """initialization of base class for forecasting based neurla anomaly detection models
        :param config: (ForecastBasedNNParams): parameters of general forecasting based neural anomaly detection models
        """
        self.model = None
        self.config = config

    def fit(
        self,
        train_data: ForecastNNVectorizedDataset,
        dev_data: ForecastNNVectorizedDataset,
    ):
        """fit method to train forecasting based neural anomaly detection models

        :param train_data: (ForecastNNVectorizedDataset): training dataset of type ForecastNNVectorizedDataset (consisting of session_idx, features, window_anomalies and window_labels)
        :param dev_data: (ForecastNNVectorizedDataset): development dataset of type ForecastNNVectorizedDataset (consisting of session_idx, features, window_anomalies and window_labels)
        """
        dataloader_train = DataLoader(
            train_data.dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=True,
        )
        dataloader_dev = DataLoader(
            dev_data.dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=True,
        )
        self.model.fit(train_loader=dataloader_train, dev_loader=dataloader_dev)

    def predict(self, test_data: ForecastNNVectorizedDataset):
        """predict method to run inference of forecasting based neural anomaly detection model on test dataset

        :param test_data: (ForecastNNVectorizedDataset): test dataset of type ForecastNNVectorizedDataset (consisting of session_idx, features, window_anomalies and window_labels)
        :return:dict: dict containing overall evaluation results
        """
        dataloader_test = DataLoader(
            test_data.dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=True,
        )
        result = self.model.predict(test_loader=dataloader_test)
        return result


@factory.register("detection", "lstm", LSTMParams)
class ForecastBasedLSTM(ForcastBasedNeuralAD):
    """Forecasting based lstm model for log anomaly detection
    """

    def __init__(self, config: LSTMParams):
        """initializing ForecastBasedLSTM object
        
        :param config: (LSTMParams): config object containing parameters for LSTM based anomaly detection model
        """
        super().__init__(config)
        self.config = config
        self.model = LSTM(config=self.config)


@factory.register("detection", "cnn", CNNParams)
class ForecastBasedCNN(ForcastBasedNeuralAD):
    """Forecasting based cnn model for log anomaly detection
    """

    def __init__(self, config: CNNParams):
        """initializing ForecastBasedCNN object
        
        :param config: (CNNParams): config object containing parameters for CNN based anomaly detection model
        """
        super().__init__(config)
        self.config = config
        self.model = CNN(config=self.config)


@factory.register("detection", "transformer", TransformerParams)
class ForecastBasedTransformer(ForcastBasedNeuralAD):
    """Forecasting based transformer model for log anomaly detection
    """

    def __init__(self, config: TransformerParams):
        """initializing ForecastBasedTransformer object
        
        :param config: (TransformerParams): config object containing parameters for Transformer based anomaly detection model.
        """
        super().__init__(config)
        self.config = config
        self.model = Transformer(config=self.config)
