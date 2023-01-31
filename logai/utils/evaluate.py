#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


def get_accuracy_precision_recall(y: np.array, y_labels: np.array):
    """
    Evaluates the anomaly and labels.

    :param y: Model inference results.
    :param y_labels: y labels.
    :return: Accuracy, precision, and recall.
    """

    if len(y) != len(y_labels):
        raise IndexError("The length of anomalies and labels should be the same")

    accuracy = accuracy_score(y, y_labels)
    precision = precision_score(y, y_labels)
    recall = recall_score(y, y_labels)
    return accuracy, precision, recall
