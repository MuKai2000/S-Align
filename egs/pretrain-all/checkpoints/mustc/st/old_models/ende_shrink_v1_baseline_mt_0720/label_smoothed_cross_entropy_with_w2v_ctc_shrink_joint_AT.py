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
from fairseq.logging.meters import safe_round

from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
import torch.nn as nn

@register_criterion("label_smoothed_cross_entropy_with_w2v_ctc_shrink_joint_AT")
class LabelSmoothedCrossEntropywithW2vCtcShrinkJointAT(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self, task, sentence_avg, label_smoothing, post_process="letter", ctc_weight=0.0, contrastive_alpha=0.0, 
                 contrastive_beta=0.0, contrastive_temperature=1.0, decrease_step=5000, get_similarity=False, is_shrink="", 
                 train_st_without_ctc=False, use_token_contrastive=False, use_two_contrastive=False, add_proj_norm=False, 
                 use_double_ctc=False, use_ctc_cluster=False, word_align=False, ban_cl_step=-1):
        super().__init__(task, sentence_avg, label_smoothing)
        self.blank_idx = task.target_dictionary.index(task.blank_symbol) if hasattr(task, 'blank_symbol') else 0
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.report_accuracy = True

        #assert 0 <= ctc_weight <= 1
        self.ctc_weight = ctc_weight
        if self.ctc_weight >= 0:
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
        self.train_st_without_ctc = train_st_without_ctc
        self.use_token_contrastive = use_token_contrastive
        self.use_two_contrastive = use_two_contrastive
        self.add_proj_norm = add_proj_norm
        self.use_double_ctc = use_double_ctc
        self.use_ctc_cluster = use_ctc_cluster
        self.word_align = word_align
        self.ban_cl_step = ban_cl_step
        self.at_level = task.at_level
        self.at_scale = task.at_scale
        self.at_nopad = task.at_nopad
        self.at_nomute = task.at_nomute
        self.at_win = {'size':20, 'stride':10}     # works when "at_level" is "window"                      # //need_check//
        self.embedding_l2norm = task.embedding_l2norm
        assert (self.at_level == "sentence" 
             or self.at_level == "token" 
             or self.at_level == "window") , "Args: --at-level is wrong, expect \"sentence\" or \"token\" or \"window\"."      # //need_check//

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
        parser.add_argument(
            "--train-st-without-ctc",
            default=False,
            type=bool,
            help="train st task without ctc loss but with constrastive loss",
        )
        parser.add_argument(
            "--use-token-contrastive",
            default=False,
            type=bool,
            help="use token level to contrastive learning",
        )
        parser.add_argument(
            "--use-two-contrastive",
            default=False,
            type=bool,
            help="use both token and sentence level to contrastive learning",
        )
        parser.add_argument(
            "--use-double-ctc",
            default=False,
            type=bool,
            help="use both pos and word level ctc",
        )
        parser.add_argument(
            "--add-proj-norm",
            default=False,
            type=bool,
            help="use proj norm to avoid over-fitting",
        )
        parser.add_argument(
            "--use-ctc-cluster",
            default=False,
            type=bool,
            help="use ctc cluster",
        )
        parser.add_argument(
            "--word-align",
            default=False,
            type=bool,
            help="use constrastive loss with word level",
        )
        parser.add_argument(
            "--ban-cl-step",
            default=-1,
            type=int,
            help="ban cl after a certain step",
        )       

    def forward(self, model, sample, per_task, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if per_task == "st":
            src_tokens, src_lengths, prev_output_tokens = sample["net_input"].values()
            #encoder_out = model.acoustic_encoder(src_tokens=src_tokens, src_lengths=src_lengths)
            #padding_mask = lengths_to_padding_mask(src_lengths)
            #print(src_tokens.shape,src_lengths.shape,padding_mask.shape)
            encoder_out = model.acoustic_encoder(src_tokens, src_lengths)
            net_output = model.decoder(
                prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
            )
            # task_prob = model.task_net(encoder_out["encoder_out"][0])

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

            #Adversarial training
            #B x 1
            """task_prob = model.task_net(encoder_out["encoder_out"][0])
            encoder_padding_mask =  encoder_out["encoder_padding_mask"][0]
            task_id = torch.zeros(encoder_padding_mask.shape[0],1)
            # print("#######################", encoder_out["encoder_out"][0].shape, encoder_out["encoder_padding_mask"][0].shape, task_prob.shape, task_id.shape)
            if torch.cuda.is_available():
                task_id = task_id.cuda()
            task_loss = F.binary_cross_entropy_with_logits(
                    task_prob.float(),
                    task_id,
                    #weight=(~acoustic_encoder_out["padding_mask"][0]).float().transpose(0,1),
                    reduction = "sum"
            )

            logging_output["task_loss"] = utils.item(task_loss.data) if reduce else task_loss.data"""

            if model.training:
                if self.ctc_weight > 0:
                    #ctc_loss = self.compute_ctc_loss(model, sample, encoder_out)
                    #logging_output["ctc_loss"] = utils.item(ctc_loss.data)
                    w2v_ctc_loss, tmp_logging_output = self.compute_ctc_loss(model, sample, encoder_out, True, self.train_st_without_ctc)
                    logging_output.update(tmp_logging_output)
                    loss = (1 - self.ctc_weight) * loss + self.ctc_weight * w2v_ctc_loss
            else:
                w2v_ctc_loss, tmp_logging_output = self.compute_ctc_loss(model, sample, encoder_out, True, self.train_st_without_ctc)
                loss = (1 - self.ctc_weight) * loss + self.ctc_weight * w2v_ctc_loss
                logging_output.update(tmp_logging_output)

            # logging_output["loss"] = utils.item(loss.data) if reduce else loss.data

            # ************/ Add L2 norm /************ #                                                         //need_check//
            if self.embedding_l2norm:
                l2_loss = model.get_acoustic_embedding_L2_norm()
                logging_output["l2_loss"] = utils.item(l2_loss.data) if reduce else l2_loss.data
                loss += l2_loss
                # print(logging_output["loss"], logging_output["l2_loss"], loss)
                # assert False
                
            logging_output["loss"] = utils.item(loss.data) if reduce else loss.data

        # ************/ Add AT per-task /************ #
        elif per_task == 'at':                  # //need_check//

            assert task.at_training, 'BUG HERE: should not do AT task!'
            
            st_prob, st_id = None, None
            mt_prob, mt_id = None, None
            # st
            if sample['st'] is not None:
                st_src_tokens, st_src_lengths = sample['st']["net_input"]['src_tokens'], sample['st']["net_input"]['src_lengths']
                st_encoder_out = model.acoustic_encoder(st_src_tokens, st_src_lengths)

                # required data
                st_encoder_output = st_encoder_out["encoder_out"][0]                    # [len, bs, dim]
                st_encoder_padding_mask = st_encoder_out["encoder_padding_mask"][0] if self.at_nomute else torch.zeros(st_encoder_out["encoder_padding_mask"][0].shape).bool().cuda()     # [bs, len]
                if self.at_level == "sentence":
                    st_prob = model.task_net(st_encoder_output, st_encoder_padding_mask if self.at_nomute else None)
                    st_id = torch.zeros(st_encoder_padding_mask.shape[0], 1)
                elif self.at_level == "token":
                    encoder_output = st_encoder_output.transpose(0,1).reshape(-1, st_encoder_output.shape[-1])   # [bs * len, dim=512]
                    encoder_padding_mask = st_encoder_padding_mask.reshape(-1)  # [bs * len]
                    encoder_output = encoder_output[~encoder_padding_mask]      # 去除mask为True
                    st_prob = model.task_net(encoder_output.unsqueeze(0))           # [1, len*bs, dim=512] --> [len*bs, 1]
                    st_id = torch.zeros(st_prob.shape[0], 1)                        # [len*bs, 1]
                elif self.at_level == "window":
                    # 计算窗口个数
                    encoder_output = st_encoder_output.transpose(0,1)   # [bs, len, dim]
                    encoder_output[st_encoder_padding_mask.unsqueeze(-1).expand(encoder_output.shape)] = 0.0    # 将padding位置设为0，避免影响mean运算
                    _shape = encoder_output.shape # [bs, len, dim]
                    # print(" > 1. ", _shape, st_encoder_output.dtype)
                    length = _shape[1]
                    num_win = max(0, (length - self.at_win['size'])) // self.at_win['stride'] + 1 + (0 if (max(0, (length - self.at_win['size'])) % self.at_win['stride']) == 0 else 1)
                    # print(" > 2. ", length, self.at_win['size'], self.at_win['stride'], num_win)
                    task_input = torch.zeros(num_win, _shape[0], _shape[-1], dtype=torch.float16).cuda()  # [num_win, bs, dim]  保存结果
                    # print(" > 3. ", task_input.shape, task_input.dtype)
                    for i in range(num_win):
                        s = i * self.at_win['stride']
                        e = min(s + self.at_win['size'], length)    # 不超过length
                        calc_output = encoder_output[:,s:e,:]   # [bs, size, dim]
                        calc_mask = st_encoder_padding_mask[:, s:e] # [bs, size]
                        calc_len = (~calc_mask).long().sum(dim=1).unsqueeze(-1) # [bs, 1]
                        calc_len[calc_len==0.0] = 1 # 特殊0值处理
                        task_input[i,:,:] = (calc_output.sum(dim=1) / calc_len)     # [bs, dim=512]
                    # remove padding vectors ::Not implemented ::Not needed for now

                    # print(" > 4. ", task_input.shape)
                    task_input = task_input.transpose(0, 1).reshape(-1, task_input.shape[-1]) # [bs*num_win, dim=512] 
                    st_prob = model.task_net(task_input.unsqueeze(0))           # [1, num_unzero, dim=512] --> [num_unzero, 1]
                    st_id = torch.zeros(st_prob.shape[0], 1)                    # [num_unzero, 1]
                else:
                    raise NotImplementedError

            # mt
            if sample['mt'] is not None:
                mt_src_tokens, mt_src_lengths = sample['mt']["net_input"]['src_tokens'], sample['mt']["net_input"]['src_lengths']
                mt_encoder_out = model.textual_encoder(mt_src_tokens, mt_src_lengths)
                
                # required data
                mt_encoder_output = mt_encoder_out["encoder_out"][0]                    # [len, bs, dim]
                mt_encoder_padding_mask = lengths_to_padding_mask(mt_src_lengths) if self.at_nopad else torch.zeros(mt_encoder_out["encoder_padding_mask"][0].shape).bool().cuda()  # [bs, len]

                if self.at_level == "sentence":
                    mt_prob = model.task_net(mt_encoder_output, mt_encoder_padding_mask if self.at_nopad else None)
                    mt_id = torch.ones(mt_encoder_padding_mask.shape[0], 1)
                elif self.at_level == "token" or self.at_level == 'window':
                    encoder_output = mt_encoder_output.transpose(0,1).reshape(-1, mt_encoder_output.shape[-1])   # [bs * len, dim=512]
                    encoder_padding_mask = mt_encoder_padding_mask.reshape(-1)  # [bs * len]
                    encoder_output = encoder_output[~encoder_padding_mask]      # 去除mask为True
                    mt_prob = model.task_net(encoder_output.unsqueeze(0))           # [1, len*bs, dim=512] --> [len*bs, 1]
                    mt_id = torch.ones(mt_prob.shape[0], 1)                         # [len*bs, 1]
                else:
                    raise NotImplementedError

            # merge st and mt
            task_prob = torch.cat((() if st_prob is None else (st_prob,)) + (() if mt_prob is None else (mt_prob,)), dim=0)
            task_id = torch.cat((() if st_id is None else (st_id,)) + (() if mt_id is None else (mt_id,)), dim=0)
            # print(" > 7. ", task_prob.shape, task_id.shape)

            # cuda
            if torch.cuda.is_available():
                task_id = task_id.cuda()

            # calculate loss
            loss, sample_size = None, None

            if self.at_level == "sentence":
                weight = torch.cat(
                        (() if sample['st'] is None else ((~st_encoder_padding_mask).long().sum(dim=1),))
                        + (() if sample['mt'] is None else ((~mt_encoder_padding_mask).long().sum(dim=1),)),
                        dim=0
                ).unsqueeze(1)                              # [bz] + [bz]  src token length
                """
                weight = torch.cat((() if sample['st'] is None else (sample['st']["target_lengths"],))  
                                    + (() if sample['mt'] is None else (sample['mt']["target_lengths"],))
                                    , dim=0).unsqueeze(1)                          # target length"""
                # bce loss
                bce_loss = F.binary_cross_entropy_with_logits(
                        task_prob.float(),
                        task_id,
                        weight=weight,
                        reduction = "mean"
                )
                sentence_sample_size = task_prob.shape[0]   # [bz + bz]
                # merge result
                loss = bce_loss if loss is None else (loss + bce_loss)
                sample_size = sentence_sample_size if sample_size is None else (sample_size + sentence_sample_size)
            elif self.at_level == "token" or self.at_level == 'window':
                mean_set = ((0 if sample['st'] is None else st_encoder_padding_mask.shape[0])
                              + (0 if sample['mt'] is None else mt_encoder_padding_mask.shape[0]))    # bz + bz  =>  num of sentence
                weight = torch.cat(
                    (() if sample['st'] is None else (torch.ones(st_prob.shape[0], dtype=torch.float16) * (self.at_win['stride'] if self.at_level == "window" else 1),))
                    + (() if sample['mt'] is None else (torch.ones(mt_prob.shape[0], dtype=torch.float16),))
                ).unsqueeze(1) / mean_set  # [l, 1]
                
                # bce loss
                bce_loss = F.binary_cross_entropy_with_logits(
                        task_prob.float(),
                        task_id,
                        weight=weight.cuda(),        # bug
                        reduction = "sum"
                )
                token_sample_size = task_prob.shape[0]
                # merge result
                loss = bce_loss if loss is None else (loss + bce_loss)
                sample_size = token_sample_size if sample_size is None else (sample_size + token_sample_size)
            else:
                raise NotImplementedError

            loss *= self.at_scale       # set scale for at loss

            # print(st_encoder_output.shape, mt_encoder_output.shape, st_prob.shape, mt_prob.shape, loss, sample_size)
            # assert False
            
            logging_output = {
                "task_loss": utils.item(loss.data) if reduce else loss.data,
                "sample_size": sample_size,
            }

        elif per_task == "mt":
            src_tokens, src_lengths, prev_output_tokens = sample["net_input"].values()
            #with open("mt_token","a")as fo:
            #    l = [str(i) for i in src_tokens.tolist()]
            #    fo.write(" ".join(l)+"\n")
            #    #_, embed_out = model.textual_encoder.forward_embedding(src_tokens)
            #    embed_out = model.textual_encoder.embed_tokens(src_tokens)
            #    l = [str(i) for i in embed_out.mean(-1).tolist()]
            #    fo.write(" ".join(l)+"\n")
            #encoder_out = model.acoustic_encoder(src_tokens=src_tokens, src_lengths=src_lengths)
            #padding_mask = lengths_to_padding_mask(src_lengths)
            #print(src_tokens.shape,src_lengths.shape,padding_mask.shape)
            #max_l = int(src_tokens.shape[1] * 0.1)
            #src_lengths += max_l
            #src_lengths += 1
            #min_l = src_lengths.min()
            #batch = src_tokens.shape[0]
            #index = (torch.rand(1).cuda() * min_l).int()
            #insert = torch.zeros([batch,1],dtype=torch.int).cuda()
            #p1,p2 = src_tokens.split([index, src_tokens.shape[1] - index],dim=-1)
            #src_tokens = torch.cat([p1, insert, p2],dim=-1)
            #for it in batch:
            #    
            #    src_tokens[it] = 
            #print(self.task.target_dictionary.string(src_tokens))
            #src_tokens = torch.where((src_tokens == 4) | (src_tokens == 5), src_tokens, 0)
            #print(src_tokens)
            encoder_out = model.textual_encoder(src_tokens, src_lengths)
            """task_prob = model.task_net(encoder_out["encoder_out"][0])
            encoder_padding_mask =  encoder_out["encoder_padding_mask"][0]
            task_id = torch.ones(encoder_padding_mask.shape[0],1)
            if torch.cuda.is_available():
                task_id = task_id.cuda()
            #print(task_id.shape,encoder_padding_mask.shape)
            task_loss = F.binary_cross_entropy_with_logits(
                    task_prob.float(),
                    task_id,
                    #weight=(~acoustic_encoder_out["padding_mask"][0]).float().transpose(0,1),
                    reduction = "sum"
            )"""
            #print(task_prob.shape)
            #with open("mt_top","a")as fo:
            #    l = [str(i) for i in src_tokens.tolist()]
            #    fo.write(" ".join(l)+"\n")
            #    #_, embed_out = model.textual_encoder.forward_embedding(src_tokens)
            #    embed_out = encoder_out["encoder_out"][0]
            #    l = [str(i) for i in embed_out.mean(-1).tolist()]
            #    fo.write(" ".join(l)+"\n")
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
            # logging_output["task_loss"] = utils.item(task_loss.data) if reduce else task_loss.data
            logging_output["loss"] = utils.item(loss.data) if reduce else loss.data

        elif per_task == "asr":
            src_tokens, src_lengths, prev_output_tokens = sample["net_input"].values()
            encoder_out = model.acoustic_encoder(src_tokens, src_lengths)
            w2v_ctc_loss, tmp_logging_output = self.compute_ctc_loss(model, sample, encoder_out,True)
            loss = w2v_ctc_loss
            sample_size = (
                sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
            )
            logging_output = {
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
            #logging_output["w2v_ctc_loss"] = utils.item(w2v_ctc_loss.data) if reduce else w2v_ctc_loss.data
            logging_output.update(tmp_logging_output)
            logging_output["loss"] = utils.item(loss.data) if reduce else loss.data
            
        else:
            raise NotImplementedError 
            
        return loss, sample_size, logging_output

    def compute_ctc_loss(self, model, sample, acoustic_encoder_out, wav_ctc=False, only_contrastive=False):
        transcript = sample["transcript"]

        non_padding_mask = ~acoustic_encoder_out["padding_mask"][0]
        input_lengths = non_padding_mask.long().sum(-1)

        pad_mask = (transcript["tokens"] != self.pad_idx) & (
                transcript["tokens"] != self.eos_idx
        )
        lprobs = model.get_acoustic_normalized_probs( acoustic_encoder_out, log_probs=True, ctc_contrastive=True).contiguous()  # (T, B, C) from the encoder
        if self.use_ctc_cluster:
            cluster_lprobs = model.get_acoustic_normalized_probs( acoustic_encoder_out, log_probs=True, cluster=True).contiguous()  # (T, B, C) from the encoder
            cluster_targets_flat = transcript["cluster_tokens"].masked_select(pad_mask)
        targets_flat = transcript["tokens"].masked_select(pad_mask)
        transcript_lengths = pad_mask.sum(-1)

        #print(targets_flat.shape,transcript["tokens"].masked_select(pad_mask).shape)
        #lp=lprobs.transpose(0, 1)
        #toks = lp.argmax(dim=-1)#.unique_consecutive()
        #with open("ctc_decode","a") as f:
        #    f.write("H- "+self.task.target_dictionary.string(toks.cpu())+"\n")
        #    f.write("R- "+self.task.target_dictionary.string(transcript["tokens"].cpu())+"\n")
        #    f.write("T- "+self.task.target_dictionary.string(sample["target"].cpu())+"\n")
        #print(toks)
        #mask_=toks>4

        if only_contrastive:
            loss = 0.0
        else:
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
            if self.use_ctc_cluster: 
                with torch.backends.cudnn.flags(enabled=False):
                    cluster_loss = F.ctc_loss(
                        cluster_lprobs,
                        cluster_targets_flat,
                        input_lengths,
                        transcript_lengths,
                        blank=self.blank_idx,
                        reduction="sum",
                        zero_infinity=self.zero_infinity,
                    )
                loss += cluster_loss
                #pred_units = self.task.target_dictionary.string(targets_flat.cpu())
                #print(pred_units)

        if self.contrastive_alpha > 0 and ( model.acoustic_encoder.num_updates < self.ban_cl_step or self.ban_cl_step == -1):

            if self.decrease_step != 0:
                contrastive_beta = max(0, self.contrastive_beta - int(model.acoustic_encoder.num_updates / self.decrease_step) * 0.1)
            else:
                contrastive_beta = self.contrastive_beta

            if contrastive_beta > 1e-4:
                reverse_sample = self.swap_sample(sample)
                if self.use_token_contrastive:
                    with torch.no_grad():
                        #embed_out = model.textual_encoder.embed_tokens(reverse_sample["net_input"]["src_tokens"])
                        _, embed_out = model.textual_encoder.forward_embedding(reverse_sample["net_input"]["src_tokens"])
                    tokens_padding_mask = reverse_sample["net_input"]["src_tokens"].eq(self.padding_idx)
                    #tokens = reverse_sample["net_input"]["src_tokens"]
                    #ctc_tokens = lprobs.transpose(0, 1).argmax(dim=-1)#.unique_consecutive()
                    #kd_sample = self.gen_kd_sample(sample, lprobs)
                    #print(tokens.shape, kd_sample["net_input"]["src_tokens"].shape)
                    contrastive_loss, similarity = self.get_token_contrastive_loss(
                        acoustic_encoder_out["wav2vec_feature"][0].transpose(0, 1),
                        acoustic_encoder_out["encoder_padding_mask"][0],
                        embed_out,
                        tokens_padding_mask
                    )
                    if self.use_two_contrastive:
                        with torch.no_grad():
                            mt_out = model.textual_encoder(reverse_sample["net_input"]["src_tokens"], reverse_sample["net_input"]["src_lengths"])
                        sentence_contrastive_loss, similarity = self.get_contrastive_loss(
                            acoustic_encoder_out,
                            mt_out,
                        )
                        contrastive_loss = sentence_contrastive_loss + contrastive_loss

                else:
                    with torch.no_grad():
                        mt_out = model.textual_encoder(reverse_sample["net_input"]["src_tokens"], reverse_sample["net_input"]["src_lengths"])
                    contrastive_loss, similarity = self.get_contrastive_loss(
                        acoustic_encoder_out,
                        mt_out,
                    )
            else:
                contrastive_loss = 0.0
                similarity = 0.0
            if contrastive_beta < 1:
                #print(sample["net_input"]["src_tokens"].shape)
                kd_sample = self.gen_kd_sample(sample, lprobs, input_lengths)
                #print(kd_sample["net_input"]["src_tokens"].shape)
                if self.use_token_contrastive:
                    with torch.no_grad():
                        _, embed_out = model.textual_encoder.forward_embedding(kd_sample["net_input"]["src_tokens"])
                    #tokens=kd_sample["net_input"]["src_tokens"]
                    tokens_padding_mask = kd_sample["net_input"]["src_tokens"].eq(self.padding_idx)
                    contrastive_kd_loss, similarity = self.get_token_contrastive_loss(
                        acoustic_encoder_out["wav2vec_feature"][0].transpose(0, 1),
                        acoustic_encoder_out["encoder_padding_mask"][0],
                        embed_out,
                        tokens_padding_mask)
                    if self.use_two_contrastive:
                        with torch.no_grad():
                            kd_mt_out = model.textual_encoder(kd_sample["net_input"]["src_tokens"], kd_sample["net_input"]["src_lengths"])
                        sentence_contrastive_kd_loss, kd_similarity = self.get_contrastive_loss(
                            acoustic_encoder_out,
                            kd_mt_out,
                            self.word_align
                        )
                        contrastive_kd_loss = sentence_contrastive_kd_loss + contrastive_kd_loss
                else:
                    with torch.no_grad():
                        kd_mt_out = model.textual_encoder(kd_sample["net_input"]["src_tokens"], kd_sample["net_input"]["src_lengths"])
                    contrastive_kd_loss, kd_similarity = self.get_contrastive_loss(
                        acoustic_encoder_out,
                        kd_mt_out,
                        self.word_align
                    )
            else:
                contrastive_kd_loss = 0.0
                kd_similarity = 0.0

            nsentences = sample["transcript"]["tokens"].size(0)
            ntokens = sample["transcript"]["ntokens"]

            if self.is_shrink == "" or self.word_align:
                all_loss = loss + (contrastive_beta * contrastive_loss + (1 - contrastive_beta) * contrastive_kd_loss) * self.contrastive_alpha / ntokens
            else:
                all_loss = loss + (contrastive_beta * contrastive_loss + (1 - contrastive_beta) * contrastive_kd_loss) * self.contrastive_alpha * ntokens / nsentences
        else:
            all_loss=loss

        if not only_contrastive:
            if wav_ctc:
                logging_output = {
                    "w2v_ctc_loss": utils.item(loss.data),  # * sample['ntokens'],
                }
            else:
                logging_output = {
                    "ctc_loss": utils.item(loss.data),  # * sample['ntokens'],
                }
        else:
            if wav_ctc:
                logging_output = {
                    "w2v_ctc_loss": utils.item(0.0),  # * sample['ntokens'],
                }
            else:
                logging_output = {
                    "ctc_loss": utils.item(0.0),  # * sample['ntokens'],
                }

        if self.contrastive_alpha > 0 and ( model.acoustic_encoder.num_updates < self.ban_cl_step or self.ban_cl_step == -1):
            logging_output["contrastive_loss"] = (contrastive_beta * contrastive_loss + (1 - contrastive_beta) * contrastive_kd_loss).data
        else:
            logging_output["contrastive_loss"] = 0 

        if not model.training:
            import editdistance

            with torch.no_grad():
                lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()

                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0
                for lp, t, ct, inp_l in zip(
                    lprobs_t,
                    transcript["tokens"],
                    transcript["cluster_tokens"]
                    if self.use_ctc_cluster
                    else transcript["tokens"],
                    input_lengths,
                ):
                    lp = lp[:inp_l].unsqueeze(0)

                    decoded = None
                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )
                    #if self.use_ctc_cluster:
                    #    targ = ct[p]
                    #else:
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_words = post_process(targ_units, self.post_process).split()
                    targ_units_arr = targ.tolist()

                    toks = lp.argmax(dim=-1).unique_consecutive()
                    pred_units_arr = toks[toks != self.blank_idx].tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    #print("ref",targ_units,)
                    #print("hyp",pred_units,"|||",lp.shape[1],"\n")
                    #tmp_toks = lp.argmax(dim=-1)
                    #tmp_words = wlp.argmax(dim=-1)
                    #print(len(targ_words))
                    #print(tmp_toks.tolist())
                    #print(tmp_words.tolist())
                    #f_toks=tmp_toks[tmp_toks != self.blank_idx].tolist()
                    #print(targ.tolist(), tmp_toks.tolist(), len(targ.tolist()), len(f_toks))
                    
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
        return all_loss, logging_output
        #else:
        #    return all_loss

    def swap_sample(self, sample):
        transcript = sample["transcript"]
        transcript_tokens = transcript["tokens"]
        transcript_lengths = transcript["lengths"] - 1
        ntokens= transcript_lengths.sum()
        target_data=sample["net_input"]["src_tokens"].contiguous()
        sample_id=sample["id"]
        #prev_output_tokens = sample["net_input"]["prev_output_tokens"]
        #src_tokens = torch.cat((prev_output_tokens[:, :1], sample["net_input"]['src_tokens']), dim=-1)
        return {
            "net_input": {
                "src_tokens": transcript_tokens.contiguous(),
                "src_lengths": transcript_lengths.contiguous(),
                "ntokens": ntokens
            },
            "target": target_data,
            "id": sample_id,
        }

    def gen_kd_sample(self, sample, lprobs, encoder_length):
        assert self.blank_idx==0, "the blank idx should be 0!"
        pred = lprobs.transpose(0,1).argmax(dim=-1)
        #print(pred.cpu().tolist())
        labels = pred.chunk(lprobs.shape[1])
        target_tokens=[]
        lengths=[]
        ntokens=0
        if self.is_shrink != "":
            for label, length in zip(labels, encoder_length):
                if "uniq" in self.is_shrink:
                    label = label[:,:length].unique_consecutive()
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

    def get_token_contrastive_loss(self, tokens_1, mask_1, tokens_2, mask_2):
        def _avg_pooling(tokens, padding_mask):
            mask=(~padding_mask).int()
            sentence_rep = (tokens * mask.unsqueeze(-1)).sum(dim=1) / mask.float().sum(dim=1).unsqueeze(-1)  # [batch, hidden_size]
            return sentence_rep
        anchor_feature = _avg_pooling(tokens_1, mask_1)
        contrast_feature = _avg_pooling(tokens_2, mask_2)
        batch_size = anchor_feature.shape[0]
        feature_dim = anchor_feature.shape[1]
        #print(anchor_feature.shape)
        #print(contrast_feature.shape)
        anchor_dot_contrast = self.similarity_function(anchor_feature.expand((batch_size, batch_size, feature_dim)),
                                                      torch.transpose(contrast_feature.expand((batch_size, batch_size, feature_dim)), 0, 1))

        loss = -nn.LogSoftmax(0)(torch.div(anchor_dot_contrast, self.contrastive_temperature)).diag().sum()
        return loss, 0

    def get_contrastive_loss(self, encoder_out1, encoder_out2, word_align=False):
        def _sentence_embedding(encoder_out, padding_mask):
            mask=(~padding_mask).int()
            encoder_output = encoder_out.transpose(0, 1)

            #if "src_tokens" in sample["net_input"]:
            #    src_tokens = sample["net_input"]["src_tokens"]
            #    mask = (src_tokens != self.padding_idx)
            encoder_embedding = (encoder_output * mask.unsqueeze(-1)).sum(dim=1) / mask.float().sum(dim=1).unsqueeze(-1)  # [batch, hidden_size]
            return encoder_embedding

        if self.is_shrink != "" and not word_align:
            encoder_embedding1 = _sentence_embedding(encoder_out1["encoder_out"][0], encoder_out1["encoder_padding_mask"][0])  # [batch, hidden_size]
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
            encoder_embedding1 = encoder_out1["encoder_out"][0]
            encoder_embedding2 = encoder_out2["encoder_out"][0]
            padding_mask = encoder_out1["encoder_padding_mask"][0]
            assert encoder_embedding1.shape == encoder_embedding2.shape
            mask=(~padding_mask).int().unsqueeze(-1).transpose(0, 1)
            encoder_embedding1 = encoder_embedding1 * mask
            encoder_embedding2 = encoder_embedding2 * mask
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

        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        task_loss = utils.item(
            sum(log.get("task_loss", 0) for log in logging_outputs)
        )
        metrics.log_scalar(
            "task_loss",
            task_loss / nsentences / math.log(2),
            nsentences,
            round=3,
        )
        contrastive_loss = utils.item(
            sum(log.get("contrastive_loss", 0) for log in logging_outputs)
        )
        metrics.log_scalar(
            "contrastive_loss",
            contrastive_loss / nsentences / math.log(2),
            nsentences,
            round=3,
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
