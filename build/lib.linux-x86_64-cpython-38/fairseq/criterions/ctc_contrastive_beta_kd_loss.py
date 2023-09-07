import math
from dataclasses import dataclass, field

from fairseq.criterions import FairseqCriterion,register_criterion
from fairseq.criterions.ctc import CtcCriterion,CtcCriterionConfig
from fairseq import metrics, utils
from fairseq.data.data_utils import post_process,collate_tokens
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask
from fairseq.logging.meters import safe_round
import torch.nn.functional as F

from collections import deque
from omegaconf import II

import torch
import torch.nn as nn

@dataclass
class CTCContrastiveBetaKDLossConfig(CtcCriterionConfig):
    sentence_avg: bool = II("optimization.sentence_avg")
    contrastive_alpha: float = field(
        default=0.0,
        metadata={"help": "The contrastive loss weight"},
    )
    contrastive_beta: float = field(
        default=0.0,
        metadata={"help": "The kd contrastive loss weight"},
    )
    decrease_step: int = field(
        default=5000,
        metadata={"help": "The number of step to descend beta weight"},
    )
    contrastive_temperature: float = field(
        default=1.0,
        metadata={"help": "The temperature to adjust contrastive loss"},
    )
    get_similarity: bool = field(
        default=False,
        metadata={"help": "To get the similarity between wav2vec and mbart"}
    )
    is_shrink: str = field(
        default="",
        metadata={"help": "To remove the  wav2vec blank output"}
    )

@register_criterion("ctc_contrastive_beta_kd_loss", dataclass=CTCContrastiveBetaKDLossConfig)
class CTCContrastiveBetaKDLoss(CtcCriterion):
    def __init__(self, cfg:CTCContrastiveBetaKDLossConfig, task:FairseqTask): #, sentence_avg, contrastive_lambda=0.0, contrastive_temperature=1.0, get_similarity=False):
        super().__init__(cfg, task)
        self.sentence_avg = cfg.sentence_avg
        self.contrastive_alpha = cfg.contrastive_alpha
        self.contrastive_beta = cfg.contrastive_beta
        self.contrastive_temperature = cfg.contrastive_temperature
        self.decrease_step = cfg.decrease_step
        self.get_similarity = cfg.get_similarity
        self.similarity_function = nn.CosineSimilarity(dim=-1)
        self.post_process = cfg.post_process
        self.is_shrink = cfg.is_shrink
    
    
    def swap_sample(self, sample):
        target = sample["target"]
        target_lengths = sample["target_lengths"]
        ntokens= sample["ntokens"]
        target_data=sample["net_input"]["source"].contiguous()
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
            target_lengths = torch.Tensor(lengths).cuda()
            target = collate_tokens(target_tokens, self.task.target_dictionary.pad())
        else:
            ntokens = pred.numel()
            target_lengths = torch.Tensor(pred.shape[1]).repeat(pred.shape[0]).cuda()
            target = pred
        ##pred_units_arr = toks[toks != self.blank_idx].tolist()
        #target = sample["target"]
        #target_lengths = sample["target_lengths"]
        #ntokens= sample["ntokens"]
        target_data=sample["net_input"]["source"].contiguous()
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
    
    def forward(self, model, sample, reduce=True):
        w2v_out = model(**sample["net_input"])
        #ctc loss
        lprobs = model.get_normalized_probs(
            w2v_out, log_probs=True, ctc_constrative=True).contiguous()
        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            non_padding_mask = ~w2v_out["padding_mask"]
            input_lengths = non_padding_mask.long().sum(-1)

        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        targets_flat = sample["target"].masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

        #all_loss=loss
        #contrastive_loss=loss
        #similarity=loss.data

        #toks = lprobs.transpose(0,1).argmax(dim=-1).chunk(lprobs.shape[0])
        #.unique_consecutive()
        #assert self.blank_idx==0, "the blank idx should be 0!"
        ##pred_units_arr = toks[toks != self.blank_idx].tolist()
        #print(toks.shape,toks)
       
        #if self.contrastive_beta > 1e-4 and (model.w2v_encoder.num_updates + 1) % (self.decrease_step + 1) == 0:
        #    self.contrastive_beta -= 0.1
        contrastive_beta = max(0, self.contrastive_beta - int(model.w2v_encoder.num_updates / self.decrease_step) * 0.1)

        if contrastive_beta > 1e-4: 
            reverse_sample = self.swap_sample(sample)
            mbart_out = model.mbart_encoder(reverse_sample["net_input"]["src_tokens"], reverse_sample["net_input"]["src_lengths"])
            contrastive_loss, similarity = self.get_contrastive_loss(
                w2v_out,
                mbart_out,
            )
        else:
            contrastive_loss = 0.0
            similarity = 0.0
        if contrastive_beta < 1:
            kd_sample = self.gen_kd_sample(sample, lprobs)
            kd_mbart_out = model.mbart_encoder(kd_sample["net_input"]["src_tokens"], kd_sample["net_input"]["src_lengths"])
            contrastive_kd_loss, kd_similarity = self.get_contrastive_loss(
                w2v_out,
                kd_mbart_out,
            )
        else:
            contrastive_kd_loss = 0.0
            kd_similarity = 0.0
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        nsentences = sample["target"].size(0)
        ntokens = sample["ntokens"]

        if self.is_shrink == "":
            all_loss = loss + (contrastive_beta * contrastive_loss + (1 - contrastive_beta)* contrastive_kd_loss) * self.contrastive_alpha 
        else:
            all_loss = loss + (contrastive_beta * contrastive_loss + (1 - contrastive_beta) * contrastive_kd_loss) * self.contrastive_alpha * ntokens / nsentences

        logging_output = {
            "loss": loss.data,
            "contrastive_loss": (contrastive_beta * contrastive_loss + (1 - contrastive_beta) * contrastive_kd_loss).data,
            #"similarity": similarity,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
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
                    if self.w2l_decoder is not None:
                        decoded = self.w2l_decoder.decode(lp)
                        if len(decoded) < 1:
                            decoded = None
                        else:
                            decoded = decoded[0]
                            if len(decoded) < 1:
                                decoded = None
                            else:
                                decoded = decoded[0]

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

        return all_loss, sample_size, logging_output
    
    
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
            encoder_embedding1 = _sentence_embedding(encoder_out1["encoder_out"], encoder_out1["padding_mask"])  # [batch, hidden_size]
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
    
    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        #similarity_sum = utils.item(
        #        sum(log.get("similarity", 0) for log in logging_outputs)
        #    )

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
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        #metrics.log_scalar("similarity", similarity_sum / nsentences )
        contrastive_loss = utils.item(
            sum(log.get("contrastive_loss", 0) for log in logging_outputs)
        )
        metrics.log_scalar(
            "contrastive_loss",
            contrastive_loss / nsentences / math.log(2),
            nsentences,
            round=3,
        )
        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
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
