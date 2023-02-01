#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from transformers import Trainer
from torch.utils.data import DataLoader, Dataset
from transformers.trainer_pt_utils import LabelSmoother, IterableDatasetShard
import torch


class Predictor(Trainer):
    """Custom Trainer object for running the inference of logBERT model for unsupervised anomaly detection.
    """

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test [`~torch.utils.data.DataLoader`]. Subclass and override this method if you want to inject some custom behavior.

        :param test_dataset: (`torch.utils.data.Dataset`, *optional*): The test dataset to use. If it is an `datasets.Dataset`, 
            columns not accepted by the `model.forward()` method are automatically removed. It must implement `__len__`.
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


class PredictionLabelSmoother(LabelSmoother):
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    :param epsilon: (:obj:`float`, `optional`, defaults to 0.1): The label smoothing factor.
    :param ignore_index: (:obj:`int`, `optional`, defaults to -100): The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.0
    ignore_index: int = -100
    eval_metrics_per_instance = [[], [], [], [], [], [], [], []]

    def __call__(self, model_output, labels, indices):
        logits = (
            model_output["logits"]
            if isinstance(model_output, dict)
            else model_output[0]
        )
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
        predictive_prob_top6 = torch.topk(predictive_prob, k=6, dim=1, largest=False)[
            0
        ].squeeze(dim=-1)
        predictive_logprob = log_probs.min(dim=-1)[0]
        predictive_logprob_top6 = torch.topk(predictive_logprob, k=6, dim=1)[0].squeeze(
            dim=-1
        )

        entropy = (probs * log_probs).sum(dim=-1)
        entropy_top6 = torch.topk(entropy, k=6, dim=1)[0].squeeze(dim=-1)

        nll_max_loss_indiv = nll_loss.max(dim=1)[0].squeeze(dim=-1)
        nll_sum_loss_indiv = nll_loss.sum(dim=1).squeeze(dim=-1)
        nll_num_loss_indiv = (~padding_mask).sum(dim=1).squeeze(dim=-1)
        nll_top6_loss_indiv = torch.topk(nll_loss, k=6, dim=1)[0].squeeze(dim=-1)

        self.eval_metrics_per_instance[0].extend(indices.cpu().data.numpy().tolist())
        self.eval_metrics_per_instance[1].extend(
            nll_max_loss_indiv.cpu().data.numpy().tolist()
        )
        self.eval_metrics_per_instance[2].extend(
            nll_sum_loss_indiv.cpu().data.numpy().tolist()
        )
        self.eval_metrics_per_instance[3].extend(
            nll_num_loss_indiv.cpu().data.numpy().tolist()
        )
        self.eval_metrics_per_instance[4].extend(
            nll_top6_loss_indiv.cpu().data.numpy().tolist()
        )
        self.eval_metrics_per_instance[5].extend(
            predictive_prob_top6.cpu().data.numpy().tolist()
        )
        self.eval_metrics_per_instance[6].extend(
            predictive_logprob_top6.cpu().data.numpy().tolist()
        )
        self.eval_metrics_per_instance[7].extend(
            entropy_top6.cpu().data.numpy().tolist()
        )

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (
            num_active_elements * log_probs.shape[-1]
        )
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss
