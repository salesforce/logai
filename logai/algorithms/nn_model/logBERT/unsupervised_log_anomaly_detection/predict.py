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
from .data_utils import load_dataset
import torch
import numpy as np 
import pandas as pd 
from .train import LogBERTConfig
from .eval_utils import compute_metrics
        

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

            if self.label_smoother is not None and "labels" in inputs:
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
                loss = self.label_smoother(outputs, labels, indices)
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
    eval_metrics_per_instance = [[],[], [], [], [], [], [], []]

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
        
        self.eval_metrics_per_instance[0].extend(indices.cpu().data.numpy().tolist())
        self.eval_metrics_per_instance[1].extend(nll_max_loss_indiv.cpu().data.numpy().tolist())
        self.eval_metrics_per_instance[2].extend(nll_sum_loss_indiv.cpu().data.numpy().tolist())
        self.eval_metrics_per_instance[3].extend(nll_num_loss_indiv.cpu().data.numpy().tolist())
        self.eval_metrics_per_instance[4].extend(nll_top6_loss_indiv.cpu().data.numpy().tolist())
        self.eval_metrics_per_instance[5].extend(predictive_prob_top6.cpu().data.numpy().tolist())
        self.eval_metrics_per_instance[6].extend(predictive_logprob_top6.cpu().data.numpy().tolist())
        self.eval_metrics_per_instance[7].extend(entropy_top6.cpu().data.numpy().tolist())

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss
        

class LogBERTPredict:

    def __init__(self, config: LogBERTConfig):

        self.config = config 
        self.max_input_seq_len = self.config.tokenizer_config['max_input_seq_len']
        self.truncation = self.config.tokenizer_config['truncation']
        self.padding = self.config.tokenizer_config['padding']

        self.mask_ngram = self.config.eval_config['mask_ngram']

        self.data_column_name = self.config.data_config['data_column_name']
        self.model_dir = config.trainer_config['model_dir']
        checkpoint_dir = 'checkpoint-'+str(max([int(x.split('-')[1]) for x in os.listdir(self.model_dir)]))
        model_checkpoint = os.path.abspath(os.path.join(self.model_dir, checkpoint_dir))
        print ('Loading model from ', model_checkpoint)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_config['tokenizer_name_or_dir'], use_fast=self.config.tokenizer_config['use_fast'])
        self.model = BertForMaskedLM.from_pretrained(model_checkpoint)
        self.model.tokenizer = self.tokenizer

        special_tokens = self.config.tokenizer_config['custom_tokens']
        special_tokens.extend([self.tokenizer.mask_token, self.tokenizer.sep_token, self.tokenizer.pad_token, self.tokenizer.unk_token, self.tokenizer.cls_token])
        ignore_tokens = [ ".", "*", ":", "$", "_", "-", "/"]
        special_tokens.extend(ignore_tokens)
        self.special_tokens = special_tokens
        self.special_token_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in special_tokens]
        self.mask_id = self.tokenizer.convert_tokens_to_ids("[MASK]")

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=self.padding, max_length=self.max_input_seq_len)

        print ("initialized data collator")
        training_args = TrainingArguments(
            self.model_dir,
            per_device_eval_batch_size=self.config.eval_config['per_device_eval_batch_size'],
            eval_accumulation_steps=self.config.eval_config['eval_accumulation_steps'],
            resume_from_checkpoint=True
        )

        label_smoother = CustomLabelSmoother(epsilon=training_args.label_smoothing_factor)

        self.trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=None,
            eval_dataset=None,
            data_collator=data_collator
        )

        self.trainer.label_smoother = label_smoother


    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.data_column_name], truncation=self.truncation, padding=self.padding, max_length=self.max_input_seq_len)

    def generate_masked_input(self, examples, indices):
        input_ids = examples['input_ids'][0]
        attention_masks = examples['attention_mask'][0]
        token_type_ids = examples['token_type_ids'][0]
        index = indices[0]
        input_ids = np.array(input_ids)
        #diag = np.eye(input_ids.shape[0]).astype(np.int64)

        sliding_window_diag = np.eye(input_ids.shape[0])
        mask = 1 - np.isin(input_ids, self.special_token_ids).astype(np.int64)
        sliding_window_diag = sliding_window_diag * mask 
        sliding_window_diag = sliding_window_diag[~np.all(sliding_window_diag == 0, axis=1)]
        num_sections = int(sliding_window_diag.shape[0]/self.mask_ngram)
        if num_sections <=0:
            num_sections = sliding_window_diag.shape[0]
        sliding_window_diag = np.array_split(sliding_window_diag, num_sections, axis=0)
        diag = np.array([np.sum(di, axis=0) for di in sliding_window_diag])
        
        input_rpt = np.tile(input_ids, (diag.shape[0],1))
        labels = np.copy(input_rpt)
        input_ids_masked = (input_rpt * (1-diag) + diag * self.mask_id).astype(np.int64)
        attention_masks = np.tile(np.array(attention_masks), (input_ids_masked.shape[0],1))
        token_type_ids = np.tile(np.array(token_type_ids), (input_ids_masked.shape[0],1))
        labels[input_ids_masked!=self.mask_id] = -100  # Need masked LM loss only for tokens with mask_id 
        examples['input_ids'] = input_ids_masked
        examples['attention_mask'] = attention_masks
        examples['token_type_ids'] = token_type_ids
        examples['labels'] = labels
        examples['indices'] = np.array([index]*input_ids_masked.shape[0]).astype(np.int64)
        return examples
        
    def predict(self, test_file, output_filename):
        test_dataset, test_labels, test_counts = load_dataset(test_file, self.data_column_name, self.special_tokens)
        print (test_dataset[1])
        test_tokenized_datasets = test_dataset.map(self.tokenize_function, batched=True, num_proc=1, remove_columns=[self.data_column_name])
        test_masked_lm_dataset = test_tokenized_datasets.map(self.generate_masked_input, with_indices=True, batched=True, batch_size=1, num_proc=1)
        
        num_shards = 100
        for i in range(num_shards):
            test_masked_lm_shard = test_masked_lm_dataset.shard(num_shards=num_shards, index=i)
            test_results = self.trainer.predict(test_masked_lm_shard)
            print ("test_loss: ",test_results.metrics['test_loss'], "test_runtime: ",test_results.metrics['test_runtime'], "test_samples/s: ", test_results.metrics['test_samples_per_second'])
           
            eval_metrics_per_instance_series = pd.DataFrame(np.transpose(np.array(self.trainer.label_smoother.eval_metrics_per_instance)), index=range(len(self.trainer.label_smoother.eval_metrics_per_instance[0])), columns=['indices', 'max_loss', 'sum_loss', 'num_loss', 'top6_loss', 'top6_max_prob', 'top6_min_logprob', 'top6_max_entropy'])
            print ("number of original test instances ", len(eval_metrics_per_instance_series.groupby('indices')))

            if i%2==0 and test_labels is not None:
                compute_metrics(eval_metrics_per_instance_series, test_labels, test_counts)
        eval_metrics_per_instance_series.to_csv(output_filename)



    