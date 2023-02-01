#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import os
import time
import torch
import logging
import numpy as np
import pandas as pd
from torch import nn
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from .utils import set_device, tensor2flatten_arr
from logai.config_interfaces import Config
from logai.utils.file_utils import read_file
from attr import dataclass
from logai.algorithms.vectorization_algo.forecast_nn import ForecastNNVectorizedDataset
from torch.utils.data import DataLoader


@dataclass
class ForecastBasedNNParams(Config):
    """
    Config for neural representation learning for logs using forecasting based self-supervised tasks.

    :param model_name: name of the model.
    :param metadata_filepath: path to file containing meta data (pretrained token embeddings in case if
        semantic log representations are used in feature type).
    :param output_dir: path to output directory where the model would be dumped.
    :param feature_type: (should be "semantics" or "sequential")type of log feature representations used
        for the log-lines or log-sequences.
    :param label_type: type of label (should be "anomaly" or "next_log") based on whether supervised
        or unsupervised (forcast based) model is being used.
    :param eval_type: (should be "session" or None) whether to aggregate and report the evaluation
        metrics at the level of sessions (based on the span_id in the log data) or at the level of each logline.
    :param topk: the prediction at top-k to consider, when deciding whether an evaluation instance is an anomaly or not.
    :param embedding_dim: dimension of the embedding space. Both for sequential and semantic type feature representation,
        the input log feature representation is passed through an embedding layer which projects it to the embedding_dim.
    :param hidden_size: dimension of the hidden representations.
    :param freeze: whether to freeze the embedding layer to use the pretrained embeddings or to further train it on the given task.
    :param gpu: device number if gpu is used (otherwise -1 or None will use cpu).
    :param patience: number of eval_steps, the model waits for performance on validation data to improve, before early stopping the training.
    :param num_train_epochs: number of training epochs.
    :param batch_size: batch size.
    :param learning_rate: learning rate.
    """

    model_name: str = None
    metadata_filepath: str = None
    output_dir: str = None
    feature_type: str = ""  # sequential, semantics
    label_type: str = ""  # anomaly, next_log
    eval_type: str = "session"  # session, None
    topk: int = 10
    embedding_dim: int = 100
    hidden_size: int = 100
    freeze: bool = False
    gpu: int = None
    patience: int = 5
    num_train_epochs: int = 100
    batch_size: int = 1024
    learning_rate: int = 1e-4


class Embedder(nn.Module):
    """Learnable embedder for embedding loglines.

    :param vocab_size: vocabulary size.
    :param embedding_dim: embedding dimension.
    :param pretrain_matrix: torch.Tensor object containing the pretrained embedding of the vocabulary tokens.
    :param freeze: Freeze embeddings to pretrained ones if set to True, otherwise makes the embeddings learnable.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        pretrain_matrix: np.array = None,
        freeze: bool = False,
    ):
        super(Embedder, self).__init__()
        if pretrain_matrix is not None:
            self.embedding_layer = nn.Embedding.from_pretrained(
                pretrain_matrix, padding_idx=1, freeze=freeze
            )
        else:
            self.embedding_layer = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=1
            )

    def forward(self, x):
        return self.embedding_layer(x.long())


class ForecastBasedNN(nn.Module):
    """
    Model for learning log representations through a forecasting based self-supervised task.

    :param config: ForecastBasedNNParams config class for parameters of forecasting based neural log representation models.
    """

    def __init__(self, config: ForecastBasedNNParams):

        super(ForecastBasedNN, self).__init__()
        self.config = config
        self.device = set_device(self.config.gpu)
        self.topk = self.config.topk
        self.feature_type = self.config.feature_type
        self.label_type = self.config.label_type
        self.eval_type = self.config.eval_type
        self.patience = self.config.patience
        self.time_tracker = {}
        self.num_train_epochs = self.config.num_train_epochs
        self.learning_rate = self.config.learning_rate

        self.meta_data = read_file(self.config.metadata_filepath)
        model_save_dirpath = os.path.join(
            self.config.output_dir, "model_" + self.config.model_name
        )
        os.makedirs(model_save_dirpath, exist_ok=True)
        self.model_save_file = os.path.join(model_save_dirpath, "model.ckpt")
        if self.feature_type in ["sequential", "semantics"]:
            self.embedder = Embedder(
                self.meta_data["vocab_size"],
                embedding_dim=self.config.embedding_dim,
                pretrain_matrix=self.meta_data.get("pretrain_matrix", None),
                freeze=self.config.freeze,
            )
        else:
            logging.info(
                f"Unrecognized feature type, except sequentials or semantics, got {self.feature_type}"
            )

    def predict(self, test_loader: DataLoader, dtype: str = "test"):
        """
        Predict method on test data.

        :param test_loader: dataloader (torch.utils.data.DataLoader) for test (or development) dataset.
        :param dtype: can be of type "test" or "dev" based on which the predict method is called for.
        :return: dict object containing the overall evaluation metrics for test (or dev) data.
        """
        logging.info("Evaluating {} data.".format(dtype))

        if self.label_type == "next_log":
            return self.__predict_next_log(test_loader, dtype=dtype)
        elif self.label_type == "anomaly":
            return self.__predict_anomaly(test_loader, dtype=dtype)

    def __predict_anomaly(self, test_loader: DataLoader, dtype: str = "test"):

        model = self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            store_dict = defaultdict(list)
            infer_start = time.time()
            if dtype == "dev":
                epoch_loss = 0
                batch_cnt = 0
                for batch_input in test_loader:
                    batch_input = self.__input2device(batch_input)
                    return_dict = model.forward(batch_input)
                    epoch_loss += return_dict["loss"].item()
                    batch_cnt += 1
                epoch_loss = epoch_loss / batch_cnt
                logging.info("Dev Loss: {} ".format(epoch_loss))
                return {"loss": epoch_loss}
            elif dtype == "test":
                for batch_input in test_loader:
                    return_dict = model.forward(self.__input2device(batch_input))
                    y_prob, y_pred = return_dict["y_pred"].max(dim=1)
                    store_dict[ForecastNNVectorizedDataset.session_idx].extend(
                        tensor2flatten_arr(
                            batch_input[ForecastNNVectorizedDataset.session_idx]
                        )
                    )
                    store_dict[ForecastNNVectorizedDataset.window_anomalies].extend(
                        tensor2flatten_arr(
                            batch_input[ForecastNNVectorizedDataset.window_anomalies]
                        )
                    )
                    store_dict["window_preds"].extend(tensor2flatten_arr(y_pred))
                infer_end = time.time()
                logging.info(
                    "Finish inference. [{:.2f}s]".format(infer_end - infer_start)
                )
                self.time_tracker["test"] = infer_end - infer_start

                store_df = pd.DataFrame(store_dict)
                use_cols = [
                    ForecastNNVectorizedDataset.session_idx,
                    ForecastNNVectorizedDataset.window_anomalies,
                    "window_preds",
                ]
                session_df = (
                    store_df[use_cols]
                    .groupby(ForecastNNVectorizedDataset.session_idx, as_index=False)
                    .sum()
                )
                pred = (session_df[f"window_preds"] > 0).astype(int)
                y = (
                    session_df[ForecastNNVectorizedDataset.window_anomalies] > 0
                ).astype(int)

                assert len(store_dict["window_preds"]) == len(
                    test_loader.dataset
                ), "Length of predictions {} does not match length of test dataset {}".format(
                    len(store_dict["window_preds"]), len(test_loader.dataset)
                )
                eval_results = {
                    "f1": f1_score(y, pred),
                    "rc": recall_score(y, pred),
                    "pc": precision_score(y, pred),
                    "acc": accuracy_score(y, pred),
                    "pred": pred,
                    "true": y,
                }
                logging.info(
                    "Best result: f1: {} rc: {} pc: {}".format(
                        eval_results["f1"], eval_results["rc"], eval_results["pc"]
                    )
                )
                return eval_results

    def __predict_next_log(self, test_loader: DataLoader, dtype: str = "test"):
        model = self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            store_dict = defaultdict(list)
            infer_start = time.time()

            if dtype == "dev":
                epoch_loss = 0
                batch_cnt = 0
                y_true_prob_list = []
                y_true_list = []
                y_pred_list = []
                for batch_input in test_loader:
                    batch_input = self.__input2device(batch_input)
                    return_dict = model.forward(batch_input)
                    epoch_loss += return_dict["loss"].item()
                    y_pred = return_dict["y_pred"]
                    y_true = batch_input[ForecastNNVectorizedDataset.window_labels]
                    y_true_prob = torch.gather(y_pred, 1, y_true.unsqueeze(dim=-1))
                    y_pred = torch.argmax(y_pred, dim=-1)
                    y_true_prob_list.extend(y_true_prob.data.cpu().numpy())
                    y_true_list.extend(y_true.data.cpu().numpy())
                    y_pred_list.extend(y_pred.data.cpu().numpy())
                    batch_cnt += 1
                epoch_loss = epoch_loss / batch_cnt
                logging.info("Dev Loss: {}".format(epoch_loss))
                y_true_prob_list = np.array(y_true_prob_list)
                num_y = len(y_true_list)
                y_true_list = np.array(y_true_list)
                y_pred_list = np.array(y_pred_list)

                y_correct = np.sum(np.equal(y_true_list, y_pred_list, dtype=np.int))
                logging.info(
                    "Dev acc @ top-1: {}  correct: {} out of {}".format(
                        float(y_correct) / num_y, y_correct, num_y
                    )
                )
                return {"loss": epoch_loss}
            elif dtype == "test":
                for batch_input in test_loader:
                    batch_input = self.__input2device(batch_input)
                    return_dict = model.forward(batch_input)
                    y_pred = return_dict["y_pred"]
                    y_prob_topk, y_pred_topk = torch.topk(y_pred, self.topk)  # b x topk
                    store_dict[ForecastNNVectorizedDataset.session_idx].extend(
                        tensor2flatten_arr(
                            batch_input[ForecastNNVectorizedDataset.session_idx]
                        )
                    )
                    store_dict[ForecastNNVectorizedDataset.window_anomalies].extend(
                        tensor2flatten_arr(
                            batch_input[ForecastNNVectorizedDataset.window_anomalies]
                        )
                    )
                    store_dict[ForecastNNVectorizedDataset.window_labels].extend(
                        tensor2flatten_arr(
                            batch_input[ForecastNNVectorizedDataset.window_labels]
                        )
                    )
                    store_dict["x"].extend(
                        batch_input[ForecastNNVectorizedDataset.features]
                        .data.cpu()
                        .numpy()
                    )
                    store_dict["y_pred_topk"].extend(y_pred_topk.data.cpu().numpy())
                    store_dict["y_prob_topk"].extend(y_prob_topk.data.cpu().numpy())
                infer_end = time.time()
                logging.info("Finish inference. [{}s]".format(infer_end - infer_start))

                assert len(store_dict["x"]) == len(
                    test_loader.dataset
                ), "Length of predictions {} does not match length of test dataset {}".format(
                    len(store_dict["x"]), len(test_loader.dataset)
                )
                self.time_tracker["test"] = infer_end - infer_start
                store_df = pd.DataFrame(store_dict)
                best_result = None
                best_f1 = -float("inf")
                count_start = time.time()

                topkdf = pd.DataFrame(store_df["y_pred_topk"].tolist())
                logging.info("Calculating acc sum.")
                not_hit = None
                for col in sorted(topkdf.columns):
                    topk = col + 1
                    if not_hit is not None:
                        not_hit = np.logical_and(
                            not_hit,
                            np.array(
                                (
                                    topkdf[col]
                                    != store_df[
                                        ForecastNNVectorizedDataset.window_labels
                                    ]
                                ).astype(int)
                            ),
                        )
                    else:
                        not_hit = np.array(
                            (
                                topkdf[col]
                                != store_df[ForecastNNVectorizedDataset.window_labels]
                            ).astype(int)
                        )
                    store_df["window_pred_anomaly_{}".format(topk)] = not_hit

                logging.info("Finish generating store_df.")

                if self.eval_type == "session":
                    use_cols = [
                        ForecastNNVectorizedDataset.session_idx,
                        ForecastNNVectorizedDataset.window_anomalies,
                    ] + [
                        f"window_pred_anomaly_{topk}"
                        for topk in range(1, self.topk + 1)
                    ]
                    session_df = (
                        store_df[use_cols]
                        .groupby(
                            ForecastNNVectorizedDataset.session_idx, as_index=False
                        )
                        .sum()
                    )
                else:
                    session_df = store_df

                for topk in range(1, self.topk + 1):
                    pred = (session_df[f"window_pred_anomaly_{topk}"] > 0).astype(int)
                    y = (
                        session_df[ForecastNNVectorizedDataset.window_anomalies] > 0
                    ).astype(int)
                    eval_results = {
                        "f1": f1_score(y, pred),
                        "rc": recall_score(y, pred),
                        "pc": precision_score(y, pred),
                        "pred": pred,
                        "true": y,
                    }
                    if eval_results["f1"] >= best_f1:
                        best_result = eval_results
                        best_f1 = eval_results["f1"]
                count_end = time.time()
                logging.info("Finish counting [{}s]".format(count_end - count_start))
                logging.info(
                    "Best result: f1: {} rc: {} pc: {}".format(
                        best_result["f1"], best_result["rc"], best_result["pc"]
                    )
                )
                return best_result

    def __input2device(self, batch_input: dict):
        return {k: v.to(self.device) for k, v in batch_input.items()}

    def save_model(self):
        """Saving model to file as specified in config"""
        logging.info("Saving model to {}".format(self.model_save_file))
        try:
            torch.save(
                self.state_dict(),
                self.model_save_file,
                _use_new_zipfile_serialization=False,
            )
        except Exception as e:
            torch.save(self.state_dict(), self.model_save_file)

    def load_model(self, model_save_file: str = ""):
        """Loading model from file.

        :param model_save_file: path to file where model would be saved.
        """
        logging.info("Loading model from {}".format(self.model_save_file))
        self.load_state_dict(torch.load(model_save_file, map_location=self.device))

    def fit(self, train_loader: DataLoader, dev_loader: DataLoader = None):
        """
        Fit method for training model

        :param train_loader: dataloader (torch.utils.data.DataLoader) for the train dataset.
        :param dev_loader: dataloader (torch.utils.data.DataLoader) for the train dataset. Defaults to None, for which no evaluation is run.
        :return: dict containing the best loss on dev dataset.
        """
        self.to(self.device)
        logging.info(
            "Start training on {} batches with {}.".format(
                len(train_loader), self.device
            )
        )
        best_loss = float("inf")
        best_results = None
        worse_count = 0
        for epoch in range(1, self.num_train_epochs + 1):
            epoch_time_start = time.time()
            model = self.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

            batch_cnt = 0
            epoch_loss = 0
            for batch_input in train_loader:
                loss = model.forward(self.__input2device(batch_input))["loss"]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
                batch_cnt += 1
                if batch_cnt % 100 == 0:
                    logging.info(
                        "Batch {}, training loss : {}".format(
                            batch_cnt, epoch_loss / batch_cnt
                        )
                    )
            epoch_loss = epoch_loss / batch_cnt
            epoch_time_elapsed = time.time() - epoch_time_start
            logging.info(
                "Epoch {}/{}, training loss: {} [{}s]".format(
                    epoch, self.num_train_epochs, epoch_loss, epoch_time_elapsed
                )
            )
            self.time_tracker["train"] = epoch_time_elapsed

            if dev_loader is not None and (epoch % 1 == 0):
                eval_results = self.predict(dev_loader, dtype="dev")
                if eval_results["loss"] < best_loss:
                    best_loss = eval_results["loss"]
                    best_results = eval_results
                    best_results["converge"] = int(epoch)
                    self.save_model()
                    worse_count = 0
                else:
                    worse_count += 1
                    if worse_count >= self.patience:
                        logging.info("Early stop at epoch: {}".format(epoch))
                        break
        self.load_model(self.model_save_file)
        return best_results
