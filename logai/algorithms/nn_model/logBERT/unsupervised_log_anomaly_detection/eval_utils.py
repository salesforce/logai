#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np 

def plot_roc(x, y, label, y_name, x_name, fig_name):
    plt.plot(x, y, label=label)
    plt.ylabel(y_name)
    plt.xlabel(x_name)
    plt.legend(loc=4)
    plt.savefig(fig_name)

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

def plot_scores_kde(scores_pos, scores_neg, fig_name):
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True

    fig1 = plt.figure()
    sns.kdeplot(scores_pos, bw=0.5, color='blue')

    fig2 = plt.figure()
    sns.kdeplot(scores_neg, bw=0.5, color='red')

    pp = PdfPages(fig_name)
    fig_nums = plt.get_fignums()
    print ("num fig_nums ", len(fig_nums), fig_nums)
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

    fig1.clear()
    fig2.clear()
    plt.close(fig1)
    plt.close(fig2)
    plt.cla()

def compute_metrics(eval_metrics_per_instance_series, test_labels, test_counts):
    y, scores_mean, scores_max, scores_top6_mean, scores_top6_max_prob, scores_top6_min_logprob, scores_top6_max_entropy = compute_scores_stats(eval_metrics_per_instance_series, test_labels, test_counts)
    compute_auc_roc(y, scores_mean, scores_max, scores_top6_mean, scores_top6_max_prob, scores_top6_min_logprob, scores_top6_max_entropy)
    

def compute_scores_stats(loss_per_instance_series, test_labels, test_counts):
    loss_per_instance_series['indices'] = loss_per_instance_series['indices'].astype(int)
    y = []
    scores_mean = []
    scores_max = []
    scores_top6_mean = []
    scores_top6_max_prob = []
    scores_top6_min_logprob = []
    scores_top6_max_entropy = []
    for data in loss_per_instance_series.groupby('indices'):
        index = int(data[0])
        if index not in test_labels:
            print ('Cannot find index in test: ', index)
            continue
        max_losses = np.array(list(data[1]['max_loss']))
        sum_losses = sum(list(data[1]['sum_loss']))
        num_losses = sum(list(data[1]['num_loss']))
        top6_losses_mean = np.mean(np.array(sorted(np.array(data[1]['top6_loss']).flatten().tolist(), reverse=True)[:6]))
        top6_max_prob = np.mean(1.0 - np.array(sorted(np.array(data[1]['top6_max_prob']).flatten().tolist())[:6]))
        top6_min_logprob = np.mean(np.array(sorted(np.array(data[1]['top6_min_logprob']).flatten().tolist(), reverse=True)[:6]))
        top6_max_entropy = np.mean(np.array(sorted(np.array(data[1]['top6_max_entropy']).flatten().tolist(), reverse=True)[:6]))
        label = test_labels[index]
        if test_counts is not None:
            count = test_counts[index]
            y.extend([label]*count)
            if num_losses==0:
                scores_mean.extend([0.0]*count)
            else:
                scores_mean.extend([sum_losses/num_losses]*count)
            scores_max.extend([np.max(max_losses)]*count)
            scores_top6_mean.extend([top6_losses_mean]*count)
            scores_top6_max_prob.extend([top6_max_prob]*count)
            scores_top6_min_logprob.extend([top6_min_logprob]*count)
            scores_top6_max_entropy.extend([top6_max_entropy]*count)
        else:
            y.append(label)
            if num_losses==0:
                scores_mean.append(0.0)
            else:
                scores_mean.append(sum_losses/num_losses)
            scores_max.append(np.max(max_losses))
            scores_top6_mean.append(top6_losses_mean)
            scores_top6_max_prob.append(top6_max_prob)
            scores_top6_min_logprob.append(top6_min_logprob)
            scores_top6_max_entropy.append(top6_max_entropy)
    return y, scores_mean, scores_max, scores_top6_mean, scores_top6_max_prob, scores_top6_min_logprob, scores_top6_max_entropy


def compute_auc_roc(y, scores_mean, scores_max, scores_top6_mean, scores_top6_max_prob, scores_top6_min_logprob, scores_top6_max_entropy, plot_graph=False, plot_histogram=False):
    scores_mean_pos = np.array([scores_mean[i] for i in range(len(scores_mean)) if y[i]==1])
    scores_mean_neg = np.array([scores_mean[i] for i in range(len(scores_mean)) if y[i]==0])

    scores_max_pos = np.array([scores_max[i] for i in range(len(scores_max)) if y[i]==1])
    scores_max_neg = np.array([scores_max[i] for i in range(len(scores_max)) if y[i]==0])

    scores_top6_mean_pos = np.array([scores_top6_mean[i] for i in range(len(scores_top6_mean)) if y[i]==1])
    scores_top6_mean_neg = np.array([scores_top6_mean[i] for i in range(len(scores_top6_mean)) if y[i]==0])

    scores_top6_max_prob_pos = np.array([scores_top6_max_prob[i] for i in range(len(scores_top6_max_prob)) if y[i]==1])
    scores_top6_max_prob_neg = np.array([scores_top6_max_prob[i] for i in range(len(scores_top6_max_prob)) if y[i]==0])

    scores_top6_min_logprob_pos = np.array([scores_top6_min_logprob[i] for i in range(len(scores_top6_min_logprob)) if y[i]==1])
    scores_top6_min_logprob_neg = np.array([scores_top6_min_logprob[i] for i in range(len(scores_top6_min_logprob)) if y[i]==0])

    scores_top6_max_entropy_pos = np.array([scores_top6_max_entropy[i] for i in range(len(scores_top6_max_entropy)) if y[i]==1])
    scores_top6_max_entropy_neg = np.array([scores_top6_max_entropy[i] for i in range(len(scores_top6_max_entropy)) if y[i]==0])

    if scores_mean_pos.shape[0]>0:
        print ('Avg Pos Mean scores: ', np.mean(scores_mean_pos), ' std: ', np.std(scores_max_pos))
    if scores_mean_neg.shape[0]>0:
        print ('Avg Neg Mean scores: ', np.mean(scores_mean_neg), ' std: ', np.std(scores_mean_neg))

    if scores_max_pos.shape[0]>0:
        print ('Avg Pos Max scores: ', np.mean(scores_max_pos), ' std:', np.std(scores_max_pos))
    if scores_max_neg.shape[0]>0:
        print ('Avg Neg Max scores: ', np.mean(scores_max_neg), ' std: ', np.std(scores_max_neg))

    if scores_top6_mean_pos.shape[0]>0:
        print ('Avg Pos Top6 Mean scores: ', np.mean(scores_top6_mean_pos), ' std: ', np.std(scores_top6_mean_pos))
    if scores_top6_mean_neg.shape[0]>0:
        print ('Avg Neg Top6 Mean scores: ', np.mean(scores_top6_mean_neg), ' std: ', np.std(scores_top6_mean_neg))

    if scores_top6_max_entropy_pos.shape[0]>0:
        print ('Avg Pos Top6 Max Entropy scores: ', np.mean(scores_top6_max_entropy_pos), ' std: ', np.std(scores_top6_max_entropy_pos))
    if scores_top6_max_entropy_neg.shape[0]>0:
        print ('Avg Neg Top6 Max Entropy score: ', np.mean(scores_top6_max_entropy_neg), ' std: ', np.std(scores_top6_max_entropy_neg))

    if scores_top6_max_prob_pos.shape[0]>0:
        print ('Avg Pos Top6 Max Prob scores: ', 1.0 - np.mean(scores_top6_max_prob_pos), ' std: ', np.std(scores_top6_max_prob_pos))
    if scores_top6_max_prob_neg.shape[0]>0:
        print ('Avg Neg Top6 Max Prob score: ', 1.0 - np.mean(scores_top6_max_prob_neg), ' std: ', np.std(scores_top6_max_prob_neg))

    if scores_top6_min_logprob_pos.shape[0]>0:
        print ('Avg Pos Top6 Min logprob scores: ', np.mean(scores_top6_min_logprob_pos), ' std: ', np.std(scores_top6_min_logprob_pos))
    if scores_top6_min_logprob_neg.shape[0]>0:
        print ('Avg Neg Top6 Min logprob score: ', np.mean(scores_top6_min_logprob_neg), ' std: ', np.std(scores_top6_min_logprob_neg))

    #if label == 1:
    #    print ('index ', index, ' mean (max) loss over ', losses.shape[0], 'tokens: ', np.mean(losses), '(', np.max(losses),')  label: ', label)

    try:
        fpr_mean, tpr_mean, thresholds = metrics.roc_curve(y, scores_mean, pos_label=1)
        auc_mean = metrics.roc_auc_score(y, scores_mean)
        print ('auc_mean: ', auc_mean)

        fpr_max, tpr_max, thresholds = metrics.roc_curve(y, scores_max, pos_label=1)

        auc_max = metrics.roc_auc_score(y, scores_max)
        print ('auc_max: ', auc_max)

        fpr_top6_mean, tpr_top6_mean, thresholds = metrics.roc_curve(y, scores_top6_mean, pos_label=1)
        auc_top6_mean = metrics.roc_auc_score(y, scores_top6_mean)
        print ('auc_top6_mean: ', auc_top6_mean)

        fpr_top6_max_prob, tpr_top6_max_prob, thresholds = metrics.roc_curve(y, scores_top6_max_prob, pos_label=1)
        auc_top6_max_prob = metrics.roc_auc_score(y, scores_top6_max_prob)
        print ('auc_top6_max_prob: ',auc_top6_max_prob)

        fpr_top6_min_logprob, tpr_top6_min_logprob, thresholds = metrics.roc_curve(y, scores_top6_min_logprob, pos_label=1)
        auc_top6_min_logprob = metrics.roc_auc_score(y, scores_top6_min_logprob)
        print ('auc_top6_min_logprob: ',auc_top6_min_logprob)

        fpr_top6_max_ent, tpr_top6_max_ent, thresholds = metrics.roc_curve(y, scores_top6_max_entropy, pos_label=1)
        auc_top6_max_ent = metrics.roc_auc_score(y, scores_top6_max_entropy)
        print ('auc_top6_max_ent: ',auc_top6_max_ent)
    except:
        pass

    if plot_graph:
        #create ROC curve
        plot_roc(fpr_mean, tpr_mean, "AUC="+str(auc_mean), "tpr_mean", "fpr_mean", "auc_mean.png")
        plot_roc(fpr_max, tpr_max, "AUC="+str(auc_max), "tpr_max", "fpr_max", "auc_max.png")
        plot_roc(fpr_top6_mean, tpr_top6_mean, "AUC="+str(auc_top6_mean), "tpr_top6_mean", "fpr_top6_mean", "auc_top6_mean.png")
        plot_roc(fpr_top6_max_ent, tpr_top6_max_ent, "AUC="+str(auc_top6_max_ent), "tpr_top6_max_ent", "fpr_top6_max_ent", "auc_top6_max_ent.png")
        plot_roc(fpr_top6_max_prob, tpr_top6_max_prob, "AUC="+str(auc_top6_max_prob), "tpr_top6_max_prob", "fpr_top6_max_prob", "auc_top6_max_prob.png")
        plot_roc(fpr_top6_min_logprob, tpr_top6_min_logprob, "AUC="+str(auc_top6_min_logprob), "tpr_top6_min_logprob", "fpr_top6_min_logprob", "auc_top6_min_logprob.png")
        
    if plot_histogram:

        plot_scores_kde(scores_max_pos, scores_max_neg, 'scores_max_hist.pdf')
        plot_scores_kde(scores_mean_pos, scores_mean_neg, 'scores_mean_hist.pdf')
        plot_scores_kde(scores_top6_mean_pos, scores_top6_mean_neg, 'scores_top6_mean_hist.pdf')
        plot_scores_kde(scores_top6_max_prob_pos, scores_top6_max_prob_neg, 'scores_top6_max_prob_hist.pdf')
        plot_scores_kde(scores_top6_max_entropy_pos, scores_top6_max_entropy_neg, 'scores_top6_max_entropy_hist.pdf')
        plot_scores_kde(scores_top6_min_logprob_pos, scores_top6_min_logprob_neg, 'scores_top6_min_logprob_hist.pdf')