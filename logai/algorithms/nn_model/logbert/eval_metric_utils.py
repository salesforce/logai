#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import os.path

from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import logging


ground_truth, prediction_score = [], []


def __plot_roc(x, y, label, y_name, x_name, fig_name, output_dir=None):
    """Plotting roc curve.

    :param x:(np.array): array of x values.
    :param y:(np.array): array of y values.
    :param label:(np.array): array of label values.
    :param y_name:(str): y axis label.
    :param  x_name:(str): x axis label.
    :param fig_name:(str): figure name.
    """
    plt.plot(x, y, label=label)
    plt.ylabel(y_name)
    plt.xlabel(x_name)
    plt.legend(loc=4)
    if output_dir:
        plt.savefig(os.path.join(os.path.join(output_dir, fig_name)))
    else:
        plt.savefig(fig_name)
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()


def __plot_scores_kde(scores_pos, scores_neg, fig_name, output_name=None):
    """Plotting kernel density estimation of positive and negative scores.

    :param scores_pos: (np.array): array of positive scores.
    :param scores_neg:(np.array): array of negative scores.
    :param fig_name: (str): figure name.
    """
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True

    fig1 = plt.figure()
    sns.kdeplot(scores_pos, bw=0.5, color="blue")

    fig2 = plt.figure()
    sns.kdeplot(scores_neg, bw=0.5, color="red")
    if output_name:
        pp = PdfPages(os.path.join(output_name, fig_name))
    else:
        pp = PdfPages(fig_name)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format="pdf")
    pp.close()

    fig1.clear()
    fig2.clear()
    plt.close(fig1)
    plt.close(fig2)
    plt.cla()


def compute_metrics(eval_metrics_per_instance_series, test_labels, test_counts=None, output_dir=None):
    """Computing evaluation metric scores for anomaly detection.

    :param eval_metrics_per_instance_series:(dict): dict object consisting
        of eval metrics for each instance index.
    :param test_labels:(dict): gold labels for each instance index.
    :param test_counts:(dict): counts of each instance index.
    :raises: Exception: IndexError if the indices of eval_metrics_per_instance_series
        do not match with indices of test_labels.
    :return: list of tuples containing labels and scores computed for each index.
        - y: list of anomaly label for each instance.
        - loss_mean: list of mean loss (over all masked non-padded tokens) for each instance.
        - loss_max: list of max loss (over all masked non-padded tokens) for each instance.
        - loss_top6_mean: list of mean loss (averaged over top-k masked non-padded tokens) for each
        instance, k = 6(following LanoBERT paper https://arxiv.org/pdf/2111.09564.pdf).
        - scores_top6_max_prob: for each instance, we take the max prob. score obtained and average
        over the top-k masked (non-padded) token prediction, k = 6.
        - scores_top6_min_logprob: for each instance, we take the min logprob score obtained and average
        over the top-k masked (non-padded) token prediction, k = 6.
        - scores_top6_max_entropy: for each instance we take the max entropy score obtained and average
        over the top-k masked (non-padded) token prediction, k = 6.
    """
    eval_metrics_per_instance_series["indices"] = eval_metrics_per_instance_series[
        "indices"
    ].astype(int)
    y = []
    loss_mean = []
    loss_max = []
    loss_top6_mean = []
    scores_top6_max_prob = []
    scores_top6_min_logprob = []
    scores_top6_max_entropy = []
    for data in eval_metrics_per_instance_series.groupby("indices"):
        index = int(data[0])
        if index not in test_labels:
            raise Exception("Cannot find index in test ", index)
        max_losses = np.array(list(data[1]["max_loss"]))
        sum_losses = sum(list(data[1]["sum_loss"]))
        num_losses = sum(list(data[1]["num_loss"]))
        top6_losses_mean = np.mean(
            np.array(
                sorted(np.array(data[1]["top6_loss"]).flatten().tolist(), reverse=True)[
                    :6
                ]
            )
        )
        top6_max_prob = np.mean(
            1.0
            - np.array(
                sorted(np.array(data[1]["top6_max_prob"]).flatten().tolist())[:6]
            )
        )
        top6_min_logprob = np.mean(
            np.array(
                sorted(
                    np.array(data[1]["top6_min_logprob"]).flatten().tolist(),
                    reverse=True,
                )[:6]
            )
        )
        top6_max_entropy = np.mean(
            np.array(
                sorted(
                    np.array(data[1]["top6_max_entropy"]).flatten().tolist(),
                    reverse=True,
                )[:6]
            )
        )
        label = test_labels[index]
        if test_counts is not None:
            count = test_counts[index]
            y.extend([label] * count)
            if num_losses == 0:
                loss_mean.extend([0.0] * count)
            else:
                loss_mean.extend([sum_losses / num_losses] * count)
            loss_max.extend([np.max(max_losses)] * count)
            loss_top6_mean.extend([top6_losses_mean] * count)
            scores_top6_max_prob.extend([top6_max_prob] * count)
            scores_top6_min_logprob.extend([top6_min_logprob] * count)
            scores_top6_max_entropy.extend([top6_max_entropy] * count)
        else:
            y.append(label)
            if num_losses == 0:
                loss_mean.append(0.0)
            else:
                loss_mean.append(sum_losses / num_losses)
            loss_max.append(np.max(max_losses))
            loss_top6_mean.append(top6_losses_mean)
            scores_top6_max_prob.append(top6_max_prob)
            scores_top6_min_logprob.append(top6_min_logprob)
            scores_top6_max_entropy.append(top6_max_entropy)

    __compute_auc_roc(
        y,
        loss_mean,
        loss_max,
        loss_top6_mean,
        scores_top6_max_prob,
        scores_top6_min_logprob,
        scores_top6_max_entropy,
        output_dir=output_dir
    )

    return


def __compute_auc_roc(
    y,
    loss_mean,
    loss_max,
    loss_top6_mean,
    scores_top6_max_prob,
    scores_top6_min_logprob,
    scores_top6_max_entropy,
    plot_graph=True,
    plot_histogram=True,
    output_dir=None
):
    """Computing AUROC for each of the type of metrics

    :param y: (list): list of anomaly labels for each instances.
    :param loss_mean: (list): list of mean loss (over all masked non-padded tokens) for each instance.
    :param loss_max: (list): list of max loss (over all masked non-padded tokens) for each instance.
    :param loss_top6_mean: (list): list of mean loss (averaged over top-k masked non-padded tokens)
        for each instance, k = 6 (following LanoBERT paper https://arxiv.org/pdf/2111.09564.pdf).
    :param scores_top6_max_prob: (list): for each instance, we take the max prob. score obtained
        and average over the top-k masked (non-padded) token prediction, k = 6.
    :param scores_top6_min_logprob: (list): for each instance, we take the min logprob score obtained
        and average over the top-k masked (non-padded) token prediction, k = 6.
    :param scores_top6_max_entropy: (list): for each instance we take the max entropy score obtained
        and average over the top-k masked (non-padded) token prediction, k = 6.
    :param plot_graph: (bool, optional): whether to plot roc graph. Defaults to False.
    :param plot_histogram: (bool, optional): whether to plot scores histogram. Defaults to False.
    """

    # __compute_auc_roc_for_metric(
    #     y=y,
    #     metric=loss_mean,
    #     metric_name_str="loss_mean",
    #     plot_graph=plot_graph,
    #     plot_histogram=plot_histogram,
    # )
    # __compute_auc_roc_for_metric(
    #     y=y,
    #     metric=loss_max,
    #     metric_name_str="loss_max",
    #     plot_graph=plot_graph,
    #     plot_histogram=plot_histogram,
    # )
    # __compute_auc_roc_for_metric(
    #     y=y,
    #     metric=loss_top6_mean,
    #     metric_name_str="loss_top6_mean",
    #     plot_graph=plot_graph,
    #     plot_histogram=plot_histogram,
    # )
    compute_auc_roc_for_metric(
        y=y,
        metric=scores_top6_max_prob,
        metric_name_str="scores_top6_max_prob",
        plot_graph=plot_graph,
        plot_histogram=plot_histogram,
        output_dir=output_dir
    )  # Note that avg scores printed for this metric would be 1 - actual probability

    ground_truth.extend(y)
    prediction_score.extend(scores_top6_max_prob)

    # compute_auc_roc_for_metric(
    #     y=y,
    #     metric=scores_top6_min_logprob,
    #     metric_name_str="scores_top6_min_logprob",
    #     plot_graph=plot_graph,
    #     plot_histogram=plot_histogram,
    #     output_dir=output_dir
    # )
    # compute_auc_roc_for_metric(
    #     y=y,
    #     metric=scores_top6_max_entropy,
    #     metric_name_str="scores_top6_max_entropy",
    #     plot_graph=plot_graph,
    #     plot_histogram=plot_histogram,
    #     output_dir=output_dir
    # )


def compute_auc_roc_for_metric(
    y, metric, metric_name_str, plot_graph=True, plot_histogram=True, output_dir=None
):
    """Computing AUROC for each metric.

    :param y: (list): list of anomaly labels for each instance.
    :param metric: (list): list of metric scores for each instance.
    :param metric_name_str: (str): name of metric.
    :param plot_graph: (bool, optional): Whether to plot ROC graph. Defaults to False.
    :param plot_histogram: (bool, optional): Whether to plot histogram of metric scores. Defaults to False.
    """

    metric_pos = np.array([metric[i] for i in range(len(metric)) if y[i] == 1])
    metric_neg = np.array([metric[i] for i in range(len(metric)) if y[i] == 0])

    if metric_pos.shape[0] > 0:
        logging.info(
            "{} Pos scores:  mean: {}, std: {}".format(
                metric_name_str, np.mean(metric_pos), np.std(metric_pos)
            )
        )
    if metric_neg.shape[0] > 0:
        logging.info(
            "{} Neg scores: mean: {}, std: {}".format(
                metric_name_str, np.mean(metric_neg), np.std(metric_neg)
            )
        )

    try:
        fpr_mean, tpr_mean, thresholds = metrics.roc_curve(y, metric, pos_label=1)
        auc_mean = metrics.roc_auc_score(y, metric)
        logging.info("AUC of {}: {}".format(metric_name_str, auc_mean))
        # logging.info("fpr_mean {}: tpr_mean:{} thresholds:{}".format(fpr_mean, tpr_mean, thresholds))
        # Find the index of the threshold that maximizes the Youden's J statistic (sum of TPR and TNR minus 1)
        j_stat = tpr_mean - fpr_mean
        optimal_threshold_index = np.argmax(j_stat)
        # Get the optimal threshold
        optimal_threshold = thresholds[optimal_threshold_index]


        logging.info("Optimal Threshold:{}".format(str(optimal_threshold)))
        sensitivity, specificity, precision, f1_score, TP, FN, FP, TN = calculate_f1_score(y, metric, optimal_threshold)
        log_content = "{}:  sensitivity {}, specificity {}, precision {}, f1_score {}, TP {}, FN {}, FP {}, TN {}".format(
            metric_name_str, sensitivity, specificity, precision, f1_score, TP, FN, FP, TN)
        logging.info(log_content)
        if output_dir:
            path = os.path.join(output_dir, metric_name_str + "_threshold")
            with open(path, "a") as fp:
                fp.write(str(optimal_threshold) + "\n")

            path = os.path.join(output_dir, "f1_score")
            with open(path, "a") as fp:
                fp.write(log_content + "\n")

    except Exception as err:
        print(err)

    logging.info("\n")

    if plot_graph:
        __plot_roc(
            fpr_mean,
            tpr_mean,
            "AUC=" + str(auc_mean),
            "tpr_mean",
            "fpr_mean",
            metric_name_str + "_auc.png",
            output_dir=output_dir
        )
    if plot_histogram:
        __plot_scores_kde(metric_pos, metric_neg, metric_name_str + "_hist.pdf", output_name=output_dir)


def calculate_f1_score(true_labels, predicted_probs, threshold=0.5):
    sensitivity, specificity, precision, f1_score, TP, FN, FP, TN = 0,0,0,0,0,0,0,0
    # Convert predicted probabilities to binary predictions based on threshold
    try:
        predicted_labels = [1 if prob >= threshold else 0 for prob in predicted_probs]

        # Confusion matrix
        TP = sum((predicted_labels[i] == 1) and (true_labels[i] == 1) for i in range(len(true_labels)))
        FN = sum((predicted_labels[i] == 0) and (true_labels[i] == 1) for i in range(len(true_labels)))
        FP = sum((predicted_labels[i] == 1) and (true_labels[i] == 0) for i in range(len(true_labels)))
        TN = sum((predicted_labels[i] == 0) and (true_labels[i] == 0) for i in range(len(true_labels)))

        # Sensitivity (Recall)
        sensitivity = TP / (TP + FN)

        # Specificity
        specificity = TN / (TN + FP)

        # Precision
        precision = TP / (TP + FP)

        # F1 Score
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    except:
        pass
    return sensitivity, specificity, precision, f1_score, TP, FN, FP, TN