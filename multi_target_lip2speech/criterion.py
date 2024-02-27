# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, LabelSmoothedCrossEntropyCriterionConfig

from .helpers import SentenceProcessor

logger = logging.getLogger(__name__)


@dataclass
class MultiTargetCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    mel_weight: float = field(default=1., metadata={"help": "weight for mel loss"})


@register_criterion("multi_target", dataclass=MultiTargetCriterionConfig)
class LabelSmoothedCrossEntropyCriterionLengthMatch(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,  # true in config
        label_smoothing,
        mel_weight,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)

        self.criterion_l1 = torch.nn.L1Loss(reduction='none')  # i.e. no sum or mean reduction
        self.criterion_sc = SpectralConvergenceLoss()  # seems like sum reduction used here

        # TODO: use Cross-Entropy loss for text too - same as units?
        self.sentence_processor = SentenceProcessor()

        # reduction = sum, output losses will be summed over the batch
        reduction = 'sum'
        self.criterion_ctc = torch.nn.CTCLoss(blank=self.sentence_processor.blank, zero_infinity=True, reduction=reduction) if task.cfg.text_supervision else None
        self.ctc_weight = 1  # speech-units use this weight and also joanna hongs paper before this

        self.mel_weight = mel_weight

        self.step = 0

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        if net_output["encoder_out_mel"] is not None:
            pred, targ = net_output["encoder_out_mel"], sample["mel"]
            targ_mask = ~sample['net_input']['padding_mask'].repeat_interleave(4, dim=1)

            crop_len = min(targ_mask.sum(1).max().item(), pred.size(1), targ.size(1))

            pred = pred[:,:crop_len].contiguous()
            targ = targ[:,:crop_len].contiguous()
            targ_mask = targ_mask[:,:crop_len].contiguous()

            pred_list, targ_list = [], []
            for p, t, m in zip(pred, targ, targ_mask):
                pred_list.append(p[m])
                targ_list.append(t[m])

            if self.sentence_avg:
                # none reduction before is now summed between pred and target
                mel_loss = ((self.criterion_l1(pred, targ).mean(-1) * targ_mask).sum(1) / targ_mask.sum(1)).sum()
            else:
                mel_loss = (self.criterion_l1(pred, targ).mean(-1) * targ_mask).sum()

            # losses are summed here over pred and targets
            sc_loss = self.criterion_sc(pred_list, targ_list, self.sentence_avg)

            mel_loss += sc_loss

            loss += mel_loss * self.mel_weight

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )  # i.e. batch size
        logging_output = {
            # "loss": loss.data,
            "nll_loss": nll_loss.data,
            "mel_loss": utils.item(mel_loss.data) if net_output["encoder_out_mel"] is not None else None,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        if self.task.cfg.text_supervision:
            # log softmax for training, softmax for inference
            log_probs = torch.nn.functional.log_softmax(net_output['encoder_out_text'], dim=2)  # requires T, B, C, 2nd dim = no. of classes
            targets = sample['text_labels']  # 1D array, length = sum(text_labels_lengths)
            input_lengths = sample['input_lengths'] * 2  # remember repeat_interleave(2) used before conformer
            target_lengths = sample['text_labels_lengths']

            ctc_loss = self.criterion_ctc(log_probs, targets.cpu(), input_lengths.cpu(), target_lengths.cpu())
            loss += ctc_loss * self.ctc_weight
            logging_output['ctc_loss'] = ctc_loss.data

            if self.step % 100 == 0:
                sample_gt_text_labels_length = sample['text_labels_lengths'][0]
                sample_gt_text_labels = sample['text_labels'][:sample_gt_text_labels_length]

                # net_output['encoder_out_text'] = T, B, C
                # softmax over no. classes and select first from batch
                # use argmax for prediction over the classes
                sample_encoder_text_softmax = torch.nn.functional.softmax(net_output['encoder_out_text'], dim=2)[:, 0, :]
                sample_pred_text_labels = torch.argmax(sample_encoder_text_softmax, dim=1)

                # NOTE: during training, text labels from argmax is sparse matrix of blanks with some word tokens in between
                # as training continues, these word tokens are usually close to groundtruth text when decoded here
                # joanna hongs model predicts blanks in the same way

                sample_gt_text = self.sentence_processor.decode(sample_gt_text_labels)
                sample_pred_text = self.sentence_processor.decode(sample_pred_text_labels)

                logger.info(f'-----------------------------------------')
                logger.info((sample_encoder_text_softmax.shape, sample_pred_text_labels))
                logger.info(f'G Text: "{sample_gt_text}"')
                logger.info(f'P Text: "{sample_pred_text}"')

        logging_output['loss'] = loss.data

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        self.step += 1

        # sample size used for denominator of gradient
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        lprobs = lprobs[:, :min(lprobs.size(1), target.size(1))]
        target = target[:, :min(lprobs.size(1), target.size(1))]
        target = target.to(dtype=torch.int64)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.reshape(-1, lprobs.size(-1)), target.reshape(-1)

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        super().reduce_metrics(logging_outputs)

        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        if logging_outputs[0]["mel_loss"] is not None:
            ctc_loss_sum = sum(log.get("mel_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                "mel_loss", ctc_loss_sum / sample_size, sample_size, round=5
            )

        if 'ctc_loss' in logging_outputs[0]:
            ctc_loss_sum = sum(log.get('ctc_loss', 0) for log in logging_outputs)
            metrics.log_scalar(
                'ctc_loss', ctc_loss_sum / sample_size, sample_size, round=5
            )


class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag_list, y_mag_list, sentence_avg):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        loss = 0.
        for x_mag, y_mag in zip(x_mag_list, y_mag_list):
            loss_one_sample = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
            loss += loss_one_sample if sentence_avg else loss_one_sample * len(y_mag)
        return loss
