#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from transformers import BertForMaskedLM, DataCollatorWithPadding
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset
from transformers.trainer_pt_utils import LabelSmoother, IterableDatasetShard
import os 
from datasets import Dataset as HFDataset
from datasets import DatasetDict
import torch
import numpy as np 
import re
import pandas as pd 
from sklearn import datasets, metrics
import matplotlib.pyplot as plt
import seaborn as sns
import ast 
from matplotlib.backends.backend_pdf import PdfPages



max_index = 0

loss_per_instance = [[],[], [], [], [], [], [], []]
class CustomTrainer(Trainer):

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test [`~torch.utils.data.DataLoader`].
        Subclass and override this method if you want to inject some custom behavior.
        Args:
            test_dataset (`torch.utils.data.Dataset`, *optional*):
                The test dataset to use. If it is an `datasets.Dataset`, columns not accepted by the `model.forward()`
                method are automatically removed. It must implement `__len__`.
        """

        if isinstance(test_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                test_dataset = IterableDatasetShard(
                    test_dataset,
                    batch_size=self.args.eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                test_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
            """
            How the loss is computed by Trainer. By default, all models return the loss in the first element.
            Subclass and override for custom behavior.
            """
            
            '''print ('Input: ', tokenizer.decode(inputs['input_ids'].cpu().data.numpy().tolist()[0]))
            i = inputs['labels'].cpu().data.numpy().tolist()[0]
            i = list(filter((-100).__ne__, i))
            print ('Output: ', tokenizer.decode(i))'''

            if label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            if "indices" in inputs:
                indices = inputs.pop("indices")
            else:
                indices = None
                
            outputs = model(**inputs)
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                loss = label_smoother(outputs, labels, indices)
            else:
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
            return (loss, outputs) if return_outputs else loss


class CustomLabelSmoother(LabelSmoother):
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (:obj:`float`, `optional`, defaults to 0.1):
            The label smoothing factor.
        ignore_index (:obj:`int`, `optional`, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.0
    ignore_index: int = -100

    def __call__(self, model_output, labels, indices):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        log_probs = -torch.nn.functional.log_softmax(logits, dim=-1)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels.clamp_min_(0)
        
        nll_loss = log_probs.gather(dim=-1, index=labels)
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True)


        nll_loss.masked_fill_(padding_mask, 0.0)

        smoothed_loss.masked_fill_(padding_mask, 0.0)

        probs.masked_fill_(padding_mask, 1.0)
        log_probs.masked_fill_(padding_mask, 0.0)

        predictive_prob = probs.max(dim=-1)[0]
        predictive_prob_top6 = torch.topk(predictive_prob, k=6, dim=1, largest=False)[0].squeeze(dim=-1)
        predictive_logprob = log_probs.min(dim=-1)[0]
        predictive_logprob_top6 = torch.topk(predictive_logprob, k=6, dim=1)[0].squeeze(dim=-1)

        entropy = (probs * log_probs).sum(dim=-1)
        entropy_top6 = torch.topk(entropy, k=6, dim=1)[0].squeeze(dim=-1)

        nll_max_loss_indiv = nll_loss.max(dim=1)[0].squeeze(dim=-1)
        nll_sum_loss_indiv = nll_loss.sum(dim=1).squeeze(dim=-1)
        nll_num_loss_indiv = (~padding_mask).sum(dim=1).squeeze(dim=-1)
        nll_top6_loss_indiv = torch.topk(nll_loss, k=6, dim=1)[0].squeeze(dim=-1)

        '''index_label = [(x, test_labels[x]) for x in set(indices.cpu().data.numpy().tolist())]
        print ('index, label: ', index_label)
        print ('Original input: ',test_dataset["test"][index_label[0][0]])
        print ('nll_loss: ',nll_loss_indiv,'\n')
        #print ('probs: ', probs, '\n')'''
        
        loss_per_instance[0].extend(indices.cpu().data.numpy().tolist())
        loss_per_instance[1].extend(nll_max_loss_indiv.cpu().data.numpy().tolist())
        loss_per_instance[2].extend(nll_sum_loss_indiv.cpu().data.numpy().tolist())
        loss_per_instance[3].extend(nll_num_loss_indiv.cpu().data.numpy().tolist())
        loss_per_instance[4].extend(nll_top6_loss_indiv.cpu().data.numpy().tolist())
        loss_per_instance[5].extend(predictive_prob_top6.cpu().data.numpy().tolist())
        loss_per_instance[6].extend(predictive_logprob_top6.cpu().data.numpy().tolist())
        loss_per_instance[7].extend(entropy_top6.cpu().data.numpy().tolist())

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss


def load_dataset(data_file, data_field, special_tokens):
    #dataset = load_dataset("csv", delimiter=",", data_files={"test": test_txt_file}, column_names=['loglines'])
    data_df = pd.read_csv(data_file, encoding = 'utf8')
    data_df[data_field+'_removed_specialtokens'] = data_df[data_field].apply(lambda x: " ".join(set(x.split(' '))-set(special_tokens)))
    data_df = data_df[data_df[data_field+'_removed_specialtokens'] != ""]
    d = pd.DataFrame({data_field: list(data_df[data_field])})
    dataset = HFDataset.from_pandas(d)
    labels = {k:v for k,v in enumerate(list(data_df['labels']))}
    if 'count' in data_df:
        counts = {k:v for k,v in enumerate(list(data_df['count']))}
        print ("Total positive instances ", sum([counts[k] for k in labels if labels[k]==1]))
        print ("Total negative instances ", sum([counts[k] for k in labels if labels[k]==0]))
        counts = None
    else:
        counts = None
    return dataset, labels, counts


def tokenize_function(examples):
    return tokenizer(examples["loglines"], truncation=True, padding='max_length', max_length=block_size)

def generate_masked_input(examples, indices):
    input_ids = examples['input_ids'][0]
    attention_masks = examples['attention_mask'][0]
    token_type_ids = examples['token_type_ids'][0]
    index = indices[0]
    input_ids = np.array(input_ids)
    #diag = np.eye(input_ids.shape[0]).astype(np.int64)

    sliding_window_diag = np.eye(input_ids.shape[0])
    mask = 1 - np.isin(input_ids, special_token_ids).astype(np.int64)
    sliding_window_diag = sliding_window_diag * mask 
    sliding_window_diag = sliding_window_diag[~np.all(sliding_window_diag == 0, axis=1)]
    num_sections = int(sliding_window_diag.shape[0]/mask_ngram)
    if num_sections <=0:
        num_sections = sliding_window_diag.shape[0]
    sliding_window_diag = np.array_split(sliding_window_diag, num_sections, axis=0) 
    diag = np.array([np.sum(di, axis=0) for di in sliding_window_diag])
    
    input_rpt = np.tile(input_ids, (diag.shape[0],1))
    labels = np.copy(input_rpt)
    input_ids_masked = (input_rpt * (1-diag) + diag * mask_id).astype(np.int64)
    attention_masks = np.tile(np.array(attention_masks), (input_ids_masked.shape[0],1))
    token_type_ids = np.tile(np.array(token_type_ids), (input_ids_masked.shape[0],1))
    labels[input_ids_masked!=mask_id] = -100  # Need masked LM loss only for tokens with mask_id     
    examples['input_ids'] = input_ids_masked
    examples['attention_mask'] = attention_masks
    examples['token_type_ids'] = token_type_ids
    examples['labels'] = labels
    examples['indices'] = np.array([index]*input_ids_masked.shape[0]).astype(np.int64)
    return examples


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



def compute_scores_stats(loss_per_instance_series, test_labels):
    loss_per_instance_series['indices'] = loss_per_instance_series['indices'].astype(int)
    print("max index ", loss_per_instance_series['indices'].max())
    y = []
    scores_mean = []
    scores_max = []
    scores_top6_mean = []
    scores_top6_max_prob = []
    scores_top6_min_logprob = []
    scores_top6_max_entropy = []
    for data in loss_per_instance_series.groupby('indices'):
        index = int(data[0])
        '''
        if index not in test_labels:
            print ('Cannot find index in test: ', index)
            continue
        '''
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
        

    




if __name__ == "__main__":
    max_index = 0
    dataset_name = "Thunderbird" #HBT" #"Thunderbird" #"BGL" "HDFS"
    test_file = "../../datasets/public/"+dataset_name+"/output/nonparsed/"+dataset_name+"_test.csv"

    if dataset_name == "HDFS":
        special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[INT]", "[IP]", "[BLOCK]", "[FILE]",  ".", "*", ":", "$", "_", "-", "/"]    
        block_size = 384
        lr = 1e-5
        mask_ngram = 8
    elif dataset_name == "BGL":
        special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[HEX]", "[INT]", "[IP]", ".", "*", ":", "$", "_", "-", "/"]
        block_size = 120
        lr = 1e-5
        mask_ngram = 1
    elif dataset_name == "Thunderbird":
        special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[HEX]", "[INT]", "[IP]", "[WARNING]", ".", "*", ":", "$", "_", "-", "/"]
        block_size = 120
        lr = 1e-4
        mask_ngram = 1
    elif dataset_name == "HBT":
        special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[HEX]", "[INT]", "[IP]", "[BLOCK]", "[WARNING]", ".", "*", ":", "$", "_", "-", "/"]
        block_size = 384
        lr = 1e-5
        mask_ngram = 4

    
    model_name = "bert-base-cased"
    test_dir = os.path.dirname(test_file)

    model_dir = os.path.join(test_dir, 'models')
    model_name = model_name+'_finetuned_custom-tokenizer_lr'+str(lr)+'_maxlen'+str(block_size)
    model_dir = os.path.join(model_dir, model_name)

    if "bert-base-cased" in model_name:
        tokenizer_dir = "log-bert-cased-tokenizer"
    elif "bert-base-uncased" in model_name:
        tokenizer_dir = "log-bert-uncased-tokenizer"
    tokenizer_dir = os.path.abspath(os.path.join(test_dir, tokenizer_dir))
        
    print ("tokenizer_dir: ",tokenizer_dir)
    output_dir = os.path.join(test_dir, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = os.path.join(output_dir, model_name+'.csv')

    

    #if not os.path.exists(output_filename):# and not (os.path.exists(output_filename.replace('.csv', '_pos.csv')) and os.path.exists(output_filename.replace('.csv', '_neg.csv'))):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)

    checkpoint_dir = 'checkpoint-'+str(max([int(x.split('-')[1]) for x in os.listdir(model_dir)]))
    model_checkpoint = os.path.abspath(os.path.join(model_dir, checkpoint_dir))

    print ('Loading model from ', model_checkpoint)

    model = BertForMaskedLM.from_pretrained(model_checkpoint)
    model.tokenizer = tokenizer

    
    test_dataset, test_labels, test_counts = load_dataset(test_file, "loglines", special_tokens)
    print (test_dataset[1])

    assert len(test_labels) == len(test_dataset)
    print ("number of test instances: ", len(test_dataset))

    test_tokenized_dataset = test_dataset.map(tokenize_function, batched=True, num_proc=10, remove_columns=['loglines'])
    print ("decoded text lengths: ", set([len(x["input_ids"]) for x in test_tokenized_dataset]))

    special_token_ids = [tokenizer.convert_tokens_to_ids(x) for x in special_tokens]
    mask_id = tokenizer.convert_tokens_to_ids("[MASK]")

    test_masked_lm = test_tokenized_dataset.map(generate_masked_input, with_indices=True, batched=True, batch_size=1, num_proc=20)
    
    


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='max_length', max_length=block_size)
    #data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.0)
    print ("initialized data collator")
    training_args = TrainingArguments(
        model_dir,
        per_device_eval_batch_size=50,
        resume_from_checkpoint=True,
        eval_accumulation_steps=40
    )

    label_smoother = CustomLabelSmoother(epsilon=training_args.label_smoothing_factor)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        data_collator=data_collator
    )

    num_shards = 500
    for i in range(num_shards):
        test_shard_masked_lm = test_masked_lm.shard(num_shards=num_shards, index=i)
        
        print ("number of test instances after generating masks: ", len(test_tokenized_dataset))

        print ('max test incides: ', max(test_masked_lm['indices']), max(test_labels))
        test_results = trainer.predict(test_shard_masked_lm)
        print ("predictions: ", test_results.predictions.shape, "test_loss: ",test_results.metrics['test_loss'], "test_runtime: ",test_results.metrics['test_runtime'], "test_samples/s: ", test_results.metrics['test_samples_per_second'])
        print ("loss_per_instance ", len(loss_per_instance[0]))

        loss_per_instance_series = pd.DataFrame(np.transpose(np.array(loss_per_instance)), index=range(len(loss_per_instance[0])), columns=['indices', 'max_loss', 'sum_loss', 'num_loss', 'top6_loss', 'top6_max_prob', 'top6_min_logprob', 'top6_max_entropy'])
        print ("number of original test instances ", len(loss_per_instance_series.groupby('indices')))

        if i%2==0:
            y, scores_mean, scores_max, scores_top6_mean, scores_top6_max_prob, scores_top6_min_logprob, scores_top6_max_entropy  = compute_scores_stats(loss_per_instance_series, test_labels)
        compute_auc_roc(y, scores_mean, scores_max, scores_top6_mean, scores_top6_max_prob, scores_top6_min_logprob, scores_top6_max_entropy)
    loss_per_instance_series.to_csv(output_filename)

    '''
    elif os.path.exists(output_filename):
        loss_per_instance_series = pd.read_csv(output_filename)
        loss_per_instance_series['top6_loss'] = loss_per_instance_series['top6_loss'].apply(ast.literal_eval)
        loss_per_instance_series['top6_max_prob'] = loss_per_instance_series['top6_max_prob'].apply(ast.literal_eval)
        loss_per_instance_series['top6_min_logprob'] = loss_per_instance_series['top6_min_logprob'].apply(ast.literal_eval)
        loss_per_instance_series['top6_max_entropy'] = loss_per_instance_series['top6_max_entropy'].apply(ast.literal_eval)
        
        y, scores_mean, scores_max, scores_top6_mean, scores_top6_max_prob, scores_top6_min_logprob, scores_top6_max_entropy = compute_scores_stats(loss_per_instance_series, test_labels)
        compute_auc_roc(y, scores_mean, scores_max, scores_top6_mean, scores_top6_max_prob, scores_top6_min_logprob, scores_top6_max_entropy, plot_graph=True, plot_histogram=True)
    '''


        


