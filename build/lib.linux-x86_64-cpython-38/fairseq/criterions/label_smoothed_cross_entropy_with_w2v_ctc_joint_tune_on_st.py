# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.data.data_utils import post_process, collate_tokens
from fairseq.data.data_utils import lengths_to_padding_mask

from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
import torch.nn as nn

@register_criterion("label_smoothed_cross_entropy_with_w2v_ctc_joint_tune_on_st")
class LabelSmoothedCrossEntropywithW2vCtcJointTune(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self, task, sentence_avg, label_smoothing, post_process="letter", ctc_weight=0.0, 
                 contrastive_alpha=0.0, contrastive_beta=0.0, contrastive_temperature=1.0, decrease_step=5000, get_similarity=False, is_shrink=""):
        super().__init__(task, sentence_avg, label_smoothing)
        self.blank_idx = task.target_dictionary.index(task.blank_symbol) if hasattr(task, 'blank_symbol') else 0
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.report_accuracy = True

        assert 0 <= ctc_weight <= 1
        self.ctc_weight = ctc_weight
        if self.ctc_weight > 0:
            assert getattr(task, "src_dict", None) is not None, "CTC need a source dictionary."
            self.zero_infinity = True
            self.post_process = post_process
        self.contrastive_alpha = contrastive_alpha
        self.contrastive_beta = contrastive_beta
        self.contrastive_temperature = contrastive_temperature
        self.decrease_step = decrease_step
        self.get_similarity = get_similarity
        self.similarity_function = nn.CosineSimilarity(dim=-1)
        self.post_process = post_process
        self.is_shrink = is_shrink

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument(
            "--zero-infinity",
            default=True,
            type=bool,
            help="zero inf loss when source length <= target length",
        )
        parser.add_argument(
            "--ctc-weight",
            default=0.0,
            type=float,
            metavar="D",
            help="weight of CTC loss",
        )
        parser.add_argument(
            "--contrastive-alpha",
            type=float, 
            default=0.0,
            help="The contrastive loss weight",
        )
        parser.add_argument(
            "--contrastive-beta",
            type=float,
            default=0.0,
            help="The kd contrastive loss weight",
        )
        parser.add_argument(
            "--decrease-step",
            type=int,
            default=5000,
            help="The number of step to descend beta weight",
        )
        parser.add_argument(
            "--contrastive-temperature", 
            type=float,
            default=1.0,
            help="The temperature to adjust contrastive loss",
        )
        parser.add_argument(
            "--get-similarity", 
            type=bool,
            default=False,
            help="To get the similarity between acoustic and textual encoder",
        )
        parser.add_argument(
            "--is-shrink",
            type=str,
            default="",
            help="To remove the  wav2vec blank output",
        )
        parser.add_argument(
            "--post-process",
            default="letter",
            type=str,
            help="how to post process predictions into words. can be letter, "
            "wordpiece, BPE symbols, etc. "
            "See fairseq.data.data_utils.post_process() for full list of options",
        )

    def forward(self, model, sample, per_task, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        #print(per_task)
        #print(sample["net_input"]["src_tokens"].shape)
        if per_task == "st":
            src_tokens, src_lengths, prev_output_tokens = sample["net_input"].values()
            #encoder_out = model.acoustic_encoder(src_tokens=src_tokens, src_lengths=src_lengths)
            #padding_mask = lengths_to_padding_mask(src_lengths)
            #print(src_tokens.shape,src_lengths.shape,padding_mask.shape)
            encoder_out = model.acoustic_encoder(src_tokens, src_lengths)
            net_output = model.decoder(
                prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
            )

            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            sample_size = (
                sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
            )
            logging_output = {
                "trans_loss": utils.item(loss.data) if reduce else loss.data,
                "nll_loss": utils.item(nll_loss.data) if reduce else nll_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }

            if self.report_accuracy:
                n_correct, total = self.compute_accuracy(model, net_output, sample)
                logging_output["n_correct"] = utils.item(n_correct.data)
                logging_output["total"] = utils.item(total.data)

            if self.ctc_weight > 0:
                #ctc_loss = self.compute_ctc_loss(model, sample, encoder_out)
                #logging_output["ctc_loss"] = utils.item(ctc_loss.data)
                w2v_ctc_loss = self.compute_ctc_loss(model, sample, encoder_out,True)
                logging_output["w2v_ctc_loss"] = utils.item(w2v_ctc_loss.data)
                loss = (1 - self.ctc_weight) * loss + self.ctc_weight * w2v_ctc_loss
            logging_output["loss"] = utils.item(loss.data) if reduce else loss.data
        elif per_task == "mt":
            src_tokens, src_lengths, prev_output_tokens = sample["net_input"].values()
            #encoder_out = model.acoustic_encoder(src_tokens=src_tokens, src_lengths=src_lengths)
            #padding_mask = lengths_to_padding_mask(src_lengths)
            #print(src_tokens.shape,src_lengths.shape,padding_mask.shape)
            encoder_out = model.textual_encoder(src_tokens, src_lengths)
            net_output = model.decoder(
                prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
            )

            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            sample_size = (
                sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
            )
            logging_output = {
                "trans_loss": utils.item(loss.data) if reduce else loss.data,
                "nll_loss": utils.item(nll_loss.data) if reduce else nll_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
            logging_output["loss"] = utils.item(loss.data) if reduce else loss.data

        elif per_task == "asr":
            src_tokens, src_lengths, prev_output_tokens = sample["net_input"].values()
            encoder_out = model.acoustic_encoder(src_tokens, src_lengths)
            w2v_ctc_loss = self.compute_ctc_loss(model, sample, encoder_out,True)
            loss = w2v_ctc_loss
            sample_size = (
                sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
            )
            logging_output = {
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
            #logging_output["w2v_ctc_loss"] = utils.item(w2v_ctc_loss.data)
            
            logging_output["loss"] = utils.item(loss.data) if reduce else loss.data
            

        else:
            raise NotImplementedError 
            
        return loss, sample_size, logging_output

    def compute_ctc_loss(self, model, sample, acoustic_encoder_out, wav_ctc=False):
        transcript = sample["transcript"]
        lprobs = model.get_acoustic_normalized_probs( acoustic_encoder_out, log_probs=True,ctc_constrative=True).contiguous()  # (T, B, C) from the encoder

        non_padding_mask = ~acoustic_encoder_out["encoder_padding_mask"][0]
        input_lengths = non_padding_mask.long().sum(-1)

        pad_mask = (transcript["tokens"] != self.pad_idx) & (
                transcript["tokens"] != self.eos_idx
        )
        targets_flat = transcript["tokens"].masked_select(pad_mask)
        transcript_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                transcript_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

        if self.contrastive_alpha > 0:
            reverse_sample = self.swap_sample(sample)
            mt_out = model.textual_encoder(reverse_sample["net_input"]["src_tokens"], reverse_sample["net_input"]["src_lengths"])
            kd_sample = self.gen_kd_sample(sample, lprobs)
            kd_mt_out = model.textual_encoder(kd_sample["net_input"]["src_tokens"], kd_sample["net_input"]["src_lengths"])

            contrastive_beta = max(0, self.contrastive_beta - int(model.acoustic_encoder.num_updates / self.decrease_step) * 0.1)

            if contrastive_beta > 1e-4:
                contrastive_loss, similarity = self.get_contrastive_loss(
                    acoustic_encoder_out,
                    mt_out,
                )
            else:
                contrastive_loss = 0.0
                similarity = 0.0
            if contrastive_beta < 1:
                contrastive_kd_loss, kd_similarity = self.get_contrastive_loss(
                    acoustic_encoder_out,
                    kd_mt_out,
                )
            else:
                contrastive_kd_loss = 0.0
                kd_similarity = 0.0

            nsentences = sample["target"].size(0)
            ntokens = sample["ntokens"]
            if self.is_shrink == "":
                all_loss = loss + (contrastive_beta * contrastive_loss + (1 - contrastive_beta)* contrastive_kd_loss) * self.contrastive_alpha
            else:
                all_loss = loss + (contrastive_beta * contrastive_loss + (1 - contrastive_beta) * contrastive_kd_loss) * self.contrastive_alpha * ntokens / nsentences
        else:
            all_loss = loss
        if wav_ctc:
            logging_output = {
                "w2v_ctc_loss": utils.item(loss.data),  # * sample['ntokens'],
            }
        else:
            logging_output = {
                "ctc_loss": utils.item(loss.data),  # * sample['ntokens'],
            }

        if not model.training:
            import editdistance

            with torch.no_grad():
                lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()

                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0
                for lp, t, inp_l in zip(
                    lprobs_t,
                    sample["target_label"]
                    if "target_label" in sample
                    else sample["target"],
                    input_lengths,
                ):
                    lp = lp[:inp_l].unsqueeze(0)

                    decoded = None
                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_units_arr = targ.tolist()

                    toks = lp.argmax(dim=-1).unique_consecutive()
                    pred_units_arr = toks[toks != self.blank_idx].tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()

                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    if decoded is not None and "words" in decoded:
                        pred_words = decoded["words"]
                        w_errs += editdistance.eval(pred_words, targ_words)
                        wv_errs += editdistance.eval(pred_words_raw, targ_words)
                    else:
                        dist = editdistance.eval(pred_words_raw, targ_words)
                        w_errs += dist
                        wv_errs += dist

                    w_len += len(targ_words)

                logging_output["wv_errors"] = wv_errs
                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return all_loss

    def swap_sample(self, sample):
        target = sample["target"]
        target_lengths = sample["target_lengths"]
        ntokens= sample["ntokens"]
        target_data=sample["net_input"]["src_tokens"].contiguous()
        sample_id=sample["id"]
        #prev_output_tokens = sample["net_input"]["prev_output_tokens"]
        #src_tokens = torch.cat((prev_output_tokens[:, :1], sample["net_input"]['src_tokens']), dim=-1)
        return {
            "net_input": {
                "src_tokens": target.contiguous(),
                "src_lengths": target_lengths.contiguous(),
                "ntokens": ntokens
            },
            "target": target_data,
            "id": sample_id,
        }

    def gen_kd_sample(self, sample, lprobs):
        assert self.blank_idx==0, "the blank idx should be 0!"
        pred = lprobs.transpose(0,1).argmax(dim=-1)
        labels = pred.chunk(lprobs.shape[1])

        target_tokens=[]
        lengths=[]
        ntokens=0
        if self.is_shrink != "":
            for label in labels:
                if "uniq" in self.is_shrink:
                    label = label.unique_consecutive()
                if "blank" in self.is_shrink:
                    label = label[label!=0]
                lengths.append(label.shape[0])
                ntokens += label.shape[0]
                target_tokens.append(label)
            target_lengths = torch.Tensor(lengths)
            if torch.cuda.is_available():
                target_lengths = target_lengths.cuda()
            target = collate_tokens(target_tokens, self.task.target_dictionary.pad())
        else:
            ntokens = pred.numel()
            target_lengths = torch.Tensor(pred.shape[1]).repeat(pred.shape[0])
            if torch.cuda.is_available():
                target_lengths = target_lengths.cuda()
            target = pred
        target_data=sample["net_input"]["src_tokens"].contiguous()
        sample_id=sample["id"]
        #prev_output_tokens = sample["net_input"]["prev_output_tokens"]
        #src_tokens = torch.cat((prev_output_tokens[:, :1], sample["net_input"]['src_tokens']), dim=-1)
        return {
            "net_input": {
                "src_tokens": target.contiguous(),
                "src_lengths": target_lengths.contiguous(),
                "ntokens": ntokens
            },
            "target": target_data,
            "id": sample_id,
        }


    def get_contrastive_loss(self, encoder_out1, encoder_out2):
        def _sentence_embedding(encoder_out, padding_mask):
            mask=(~padding_mask).int()
            encoder_output = encoder_out.transpose(0, 1)

            #if "src_tokens" in sample["net_input"]:
            #    src_tokens = sample["net_input"]["src_tokens"]
            #    mask = (src_tokens != self.padding_idx)
            encoder_embedding = (encoder_output * mask.unsqueeze(-1)).sum(dim=1) / mask.float().sum(dim=1).unsqueeze(-1)  # [batch, hidden_size]
            return encoder_embedding

        if self.is_shrink != "":
            encoder_embedding1 = _sentence_embedding(encoder_out1["encoder_out"][0], encoder_out1["padding_mask"][0])  # [batch, hidden_size]
            encoder_embedding2 = _sentence_embedding(encoder_out2["encoder_out"][0], encoder_out2["encoder_padding_mask"][0])  # [batch, hidden_size]
            batch_size = encoder_embedding2.shape[0]
            feature_dim = encoder_embedding2.shape[1]
            anchor_feature = encoder_embedding1
            contrast_feature = encoder_embedding2
            if self.get_similarity:
                similarity = self.similarity_function(encoder_out1["wav2vec_out"].mean(1),encoder_embedding2).mean(-1)
                #print(encoder_out1["wav2vec_out"].mean(1).shape)
            #else:
            #    similarity = self.similarity_function(encoder_embedding1,encoder_embedding2).mean(-1)
            anchor_dot_contrast = self.similarity_function(anchor_feature.expand((batch_size, batch_size, feature_dim)),
                                                      torch.transpose(contrast_feature.expand((batch_size, batch_size, feature_dim)), 0, 1))

            loss = -nn.LogSoftmax(0)(torch.div(anchor_dot_contrast, self.contrastive_temperature)).diag().sum()
        else:
            encoder_embedding1 = encoder_out1["encoder_out"]
            encoder_embedding2 = encoder_out2["encoder_out"][0]
            batch_size = encoder_embedding2.shape[1]
            length = encoder_embedding2.shape[0]
            feature_dim = encoder_embedding2.shape[2]
            anchor_dot_contrast = self.similarity_function(encoder_embedding1.expand((length, length, batch_size, feature_dim)).transpose(0,2),
                                                           encoder_embedding2.expand((length, length, batch_size, feature_dim)).transpose(0,2))
            loss = -nn.LogSoftmax(1)(torch.div(anchor_dot_contrast, self.contrastive_temperature)).diagonal().sum()

        return loss, 0

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        trans_loss_sum = utils.item(
            sum(log.get("trans_loss", 0) for log in logging_outputs)
        )
        nll_loss_sum = utils.item(
            sum(log.get("nll_loss", 0) for log in logging_outputs)
        )
        w2v_ctc_loss_sum = utils.item(
            sum(log.get("w2v_ctc_loss", 0) for log in logging_outputs)
        )
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "trans_loss", trans_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "w2v_ctc_loss",
            w2v_ctc_loss_sum / sample_size / math.log(2),
            sample_size,
            round=3,
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
