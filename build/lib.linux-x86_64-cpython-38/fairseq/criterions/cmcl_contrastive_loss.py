import math
from dataclasses import dataclass, field

from fairseq.criterions import FairseqCriterion,register_criterion
from fairseq import metrics, utils
from fairseq.dataclass import FairseqDataclass

from collections import deque
from omegaconf import II

import torch
import torch.nn as nn

@dataclass
class CmclContrastiveLossConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    contrastive_lambda: float = field(
        default=0.0,
        metadata={"help": "The contrastive loss weight"},
    )
    contrastive_temperature: float = field(
        default=1.0,
        metadata={"help": "The temperature to adjust contrastive loss"},
    )
    get_similarity: bool = field(
        default=False,
        metadata={"help": "To get the similarity between wav2vec and mbart"}
    )

@register_criterion("cmcl_contrastive_loss", dataclass=CmclContrastiveLossConfig)
class CmclContrastiveLoss(FairseqCriterion):
    def __init__(self, task, sentence_avg, contrastive_lambda=0.0, contrastive_temperature=1.0, get_similarity=False):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.contrastive_lambda = contrastive_lambda
        self.contrastive_temperature = contrastive_temperature
        self.get_similarity = get_similarity
        self.similarity_function = nn.CosineSimilarity(dim=-1)
    
    
    def swap_sample(self, sample):
        target = sample["target"]
        target_lengths = sample["target_lengths"]
        ntokens= sample["ntokens"]
        #prev_output_tokens = sample["net_input"]["prev_output_tokens"]
        #src_tokens = torch.cat((prev_output_tokens[:, :1], sample["net_input"]['src_tokens']), dim=-1)
        return {
            "net_input": {
                "src_tokens": target.contiguous(),
                "src_lengths": target_lengths.contiguous(),
                "ntokens": ntokens
            },
            "target": sample["net_input"]["source"].contiguous(),
            "id": sample["id"],
        }
    
    def forward(self, model, sample, reduce=True):
        w2v_out = model(**sample["net_input"])
        #loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        #w2v_out = model.w2v_encoder.forward(sample["net_input"]["src_tokens"], sample["net_input"]["src_lengths"]).encoder_out
        #pad_mask = (sample["target"] != self.pad_idx) & (
        #    sample["target"] != self.eos_idx
        #)
        #targets_flat = sample["target"].masked_select(pad_mask)
        #if "target_lengths" in sample:
        #    target_lengths = sample["target_lengths"]
        #else:
        #    target_lengths = pad_mask.sum(-1)
        reverse_sample = self.swap_sample(sample)
        mbart_out = model.mbart_encoder(reverse_sample["net_input"]["src_tokens"], reverse_sample["net_input"]["src_lengths"])
        #reversed_encoder_out = model.encoder.forward(reverse_sample["net_input"]["src_tokens"], reverse_sample["net_input"]["src_lengths"]).encoder_out
        contrastive_loss,similarity = self.get_contrastive_loss(
            w2v_out,
            mbart_out,
            #sample,
            #reverse_sample,
        )
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        nsentences = sample["target"].size(0)
        ntokens = sample["ntokens"]
        #all_loss = loss + contrastive_loss * self.contrastive_lambda * ntokens / nsentences
        logging_output = {
            "loss": contrastive_loss.data,
            "similarity": similarity,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        #if isinstance(contrastive_loss, int):
        #    logging_output["contrastive_loss"] = 0
        #else:
        #    logging_output["contrastive_loss"] = utils.item(contrastive_loss.data)
        
        return contrastive_loss, sample_size, logging_output
    
    #def similarity_function(self, ):
    #    return nn.CosineSimilarity(dim=-1)
    
    def get_contrastive_loss(self, encoder_out1, encoder_out2):
        def _sentence_embedding(encoder_out, padding_mask):
            mask=(~padding_mask).int()
            encoder_output = encoder_out.transpose(0, 1)
            
            #if "src_tokens" in sample["net_input"]:
            #    src_tokens = sample["net_input"]["src_tokens"]
            #    mask = (src_tokens != self.padding_idx)
            encoder_embedding = (encoder_output * mask.unsqueeze(-1)).sum(dim=1) / mask.float().sum(dim=1).unsqueeze(-1)  # [batch, hidden_size]
            return encoder_embedding
        
        encoder_embedding1 = _sentence_embedding(encoder_out1["encoder_out"], encoder_out1["padding_mask"])  # [batch, hidden_size]
        encoder_embedding2 = _sentence_embedding(encoder_out2["encoder_out"][0], encoder_out2["encoder_padding_mask"][0])  # [batch, hidden_size]
        batch_size = encoder_embedding2.shape[0]
        feature_dim = encoder_embedding2.shape[1]
        anchor_feature = encoder_embedding1
        contrast_feature = encoder_embedding2
        if self.get_similarity:
            similarity = self.similarity_function(encoder_out1["wav2vec_out"].mean(1),encoder_embedding2).mean(-1)
            #print(encoder_out1["wav2vec_out"].mean(1).shape)
        else: 
            similarity = self.similarity_function(encoder_embedding1,encoder_embedding2).mean(-1)
        anchor_dot_contrast = self.similarity_function(anchor_feature.expand((batch_size, batch_size, feature_dim)),
                                                  torch.transpose(contrast_feature.expand((batch_size, batch_size, feature_dim)), 0, 1))
        
        loss = -nn.LogSoftmax(0)(torch.div(anchor_dot_contrast, self.contrastive_temperature)).diag().sum()
        
        return loss, similarity
    
    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        similarity_sum = utils.item(
                sum(log.get("similarity", 0) for log in logging_outputs)
            )
        metrics.log_scalar("similarity", similarity_sum / len(logging_outputs) )

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
        #contrastive_loss = utils.item(
        #    sum(log.get("contrastive_loss", 0) for log in logging_outputs)
        #)
        #metrics.log_scalar(
        #    "contrastive_loss",
        #    contrastive_loss / nsentences / math.log(2),
        #    nsentences,
        #    round=3,
        #)
