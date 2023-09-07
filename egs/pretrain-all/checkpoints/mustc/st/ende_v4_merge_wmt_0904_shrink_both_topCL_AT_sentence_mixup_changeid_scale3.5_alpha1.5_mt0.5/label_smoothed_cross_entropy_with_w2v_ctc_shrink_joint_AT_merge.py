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

@register_criterion("label_smoothed_cross_entropy_with_w2v_ctc_shrink_joint_AT_merge")
class LabelSmoothedCrossEntropywithW2vCtcShrinkJointATMerge(
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
        self.at_training = task.at_training
        self.at_level = task.at_level
        self.at_scale = task.at_scale
        self.at_low_pos = task.at_low_pos
        self.at_nopad = task.at_nopad
        self.at_nomute = task.at_nomute
        self.at_adapte_win = task.at_adapte_win
        self.mix_tag = task.mix_tag
        self.mixup_rate = task.mixup_rate
        self.mixup_change_id = task.mixup_change_id
        self.mixup_for_whole_model = task.mixup_for_whole_model
        if self.mixup_for_whole_model:
            assert self.mixup_rate < 0.0
        self.at_win = {'size':8, 'stride':5}     # works when "at_level" is "window"                      # //need_check//
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

    def forward(self, model, sample, per_task, reduce=True, merge_task=False, st_update=False, update_num=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # ************/ Merge ST&MT&AT Task /************ #
        # print(">>> Criterions File Testing !!!")
        
        # init
        logging_output = {
            "loss": 0.0,
            "trans_loss": 0.0,
            "nll_loss": 0.0,
            "ntokens": 0,
            "nsentences": 0,
            "sample_size": 0,
        }

        # five task will do here: st(no merge), mt, at, asr, st(merge:st, mt, at)
        loss, loss_at, loss_st, loss_mt = None, None, None, None
        sample_size = None
        prob_st, id_st = None, None
        prob_mt, id_mt = None, None
        prob_st_gen, id_st_gen = None, None
        prob_mt_gen, id_mt_gen = None, None

        multi_st = (per_task == 'st' and merge_task)            # doing st(merge:st, mt, at) task
        # task: st(no merge), st(merge:st, mt, at), at need to get output from model.acoustic_encoder
        if per_task == 'st' or (per_task == 'at' and sample['st'] is not None):            # sample_st --> acoustic_encoder --> encoder_out_st
            # ("-- Encode ST Task", per_task)
            sample_st = sample if per_task != 'at' else sample['st']
            src_tokens_st, src_lengths_st, prev_output_tokens_st = sample_st["net_input"]['src_tokens'], sample_st["net_input"]['src_lengths'], sample_st["net_input"]['prev_output_tokens']
            encoder_out_st = model.acoustic_encoder(src_tokens_st, src_lengths_st, mixup_rate=self.mixup_rate if ((per_task == 'at' or multi_st) and st_update) else 0.0, mixup_for_whole_model=self.mixup_for_whole_model, textual_encoder=model.textual_encoder, update_num=update_num)
            
            # task: st(no merge), st(merge:st, mt, at) need to get output from model.decoder and cal st loss
            if per_task == 'st':  # Cal ST Loss: encoder_out_st --> decoder --> net_output_st
                # print("-- Cal ST Loss", per_task)
                net_output_st = model.decoder(
                    prev_output_tokens=prev_output_tokens_st, encoder_out=encoder_out_st
                )
                loss_st, nll_loss_st = self.compute_loss(model, net_output_st, sample_st, reduce=reduce)
                sample_size_st = (
                    sample_st["target"].size(0) if self.sentence_avg else sample_st["ntokens"]
                )
                logging_output["trans_loss"] += (utils.item(loss_st.data) if reduce else loss_st.data)
                logging_output["nll_loss"] += (utils.item(nll_loss_st.data) if reduce else nll_loss_st.data)
                logging_output["ntokens"] += sample_st["ntokens"]
                logging_output["nsentences"] += sample_st["target"].size(0)
                logging_output["sample_size"] += sample_size_st

                if self.report_accuracy:
                    n_correct_st, total_st = self.compute_accuracy(model, net_output_st, sample_st)
                    logging_output["n_correct"] = utils.item(n_correct_st.data)
                    logging_output["total"] = utils.item(total_st.data)

                if model.training:
                    if self.ctc_weight > 0:
                        w2v_ctc_loss, tmp_logging_output = self.compute_ctc_loss(model, sample_st, encoder_out_st, True, self.train_st_without_ctc)
                        logging_output.update(tmp_logging_output)
                        loss_st = (1 - self.ctc_weight) * loss_st + self.ctc_weight * w2v_ctc_loss
                else:
                    w2v_ctc_loss, tmp_logging_output = self.compute_ctc_loss(model, sample_st, encoder_out_st, True, self.train_st_without_ctc)
                    loss_st = (1 - self.ctc_weight) * loss_st + self.ctc_weight * w2v_ctc_loss
                    logging_output.update(tmp_logging_output)

                logging_output["loss"] += utils.item(loss_st.data) if reduce else loss_st.data
                # print(f"ST Loss {per_task}: {loss_st}, Sample size {per_task}: {sample_size_st}")
                sample_size = sample_size_st if sample_size is None else sample_size + sample_size_st
                loss = loss_st if loss is None else loss + loss_st
            
            # when use self.at_training, task: at, st(merge:st, mt, at) need to get st output from model.task_net
            if self.at_training and (per_task == 'at' or multi_st):
                # print("-- Record AT data for ST", per_task)
                # required data
                if not self.at_low_pos:
                    encoder_output_st = encoder_out_st['encoder_output_mixup'][0] if (encoder_out_st['encoder_output_mixup'][0] is not None and not self.mixup_for_whole_model) else encoder_out_st["encoder_out"][0]                    # [len, bs, dim]
                    encoder_padding_mask_st = encoder_out_st["encoder_padding_mask"][0] if self.at_nomute else torch.zeros(encoder_out_st["encoder_padding_mask"][0].shape).bool().to(encoder_out_st["encoder_padding_mask"][0].device)     # [bs, len]
                else:
                    encoder_output_st = encoder_out_st['wav2vec_feature'][0]                    # [len, bs, dim]
                    encoder_padding_mask_st = torch.zeros(encoder_output_st.shape[:2]).bool().to(encoder_output_st.device)  # [bs, len]
                if self.at_level == "sentence":
                    # dis loss
                    encoder_output_st_copy = encoder_output_st.clone().detach_()
                    prob_st = model.task_net(encoder_output_st_copy, encoder_padding_mask_st if self.at_nomute else None)
                    if self.mixup_rate < 0.0 and encoder_out_st["mixup_rate"][0] is not None and self.mixup_change_id:
                        id_st = encoder_out_st["mixup_rate"][0].to(prob_st.device)         # note kkq change here 08.09.2023
                    else:
                        id_st = torch.zeros(encoder_padding_mask_st.shape[0], 1).to(prob_st.device)
                    # gen loss
                    prob_st_gen = model.task_net(encoder_output_st, encoder_padding_mask_st if self.at_nomute else None)
                    id_st_gen = torch.full((encoder_padding_mask_st.shape[0], 1), self.mix_tag).to(prob_st_gen.device)

                elif self.at_level == "token":
                    encoder_output = encoder_output_st.transpose(0,1).reshape(-1, encoder_output_st.shape[-1])   # [bs * len, dim=512]
                    encoder_padding_mask = encoder_padding_mask_st.reshape(-1)      # [bs * len]
                    encoder_output = encoder_output[~encoder_padding_mask]          # 去除mask为True
                    prob_st = model.task_net(encoder_output.unsqueeze(0))           # [1, len*bs, dim=512] --> [len*bs, 1]
                    id_st = torch.zeros(prob_st.shape[0], 1).to(prob_st.device)                        # [len*bs, 1]
                elif self.at_level == "window":
                    encoder_output = encoder_output_st.transpose(0,1)   # [bs, len, dim]
                    if self.at_nomute:
                        encoder_output[encoder_padding_mask_st.unsqueeze(-1).expand(encoder_output.shape)] = 0.0    # 将padding为True的位置设为0，避免影响mean运算
                    _shape = encoder_output.shape # [bs, len, dim]
                    length = _shape[1]
                    num_win = max(0, (length - self.at_win['size'])) // self.at_win['stride'] + 1 + (0 if (max(0, (length - self.at_win['size'])) % self.at_win['stride']) == 0 else 1)
                    # print(" > 2. ", length, tgt_len, self.at_win['size'], self.at_win['stride'], num_win)
                    task_input = torch.zeros(num_win, _shape[0], _shape[-1], dtype=torch.float16).to(encoder_output.device)  # [num_win, bs, dim]  to save result
                    # print(" > 3. ", task_input.shape, task_input.dtype)
                    for i in range(num_win):
                        s = i * self.at_win['stride']
                        e = min(s + self.at_win['size'], length)    # 不超过length
                        calc_output = encoder_output[:,s:e,:]   # [bs, size, dim]
                        if self.at_nomute:
                            calc_mask = encoder_padding_mask_st[:, s:e] # [bs, size]
                            calc_len = (~calc_mask).long().sum(dim=1).unsqueeze(-1) # [bs, 1]
                            calc_len[calc_len==0.0] = 1 # 特殊0值处理
                            task_input[i,:,:] = (calc_output.sum(dim=1) / calc_len)     # [bs, dim=512]
                        else:
                            task_input[i,:,:] = calc_output.mean(dim=1)
                    
                    # print(" > 4. ", task_input.shape)
                    task_input = task_input.transpose(0, 1).reshape(-1, task_input.shape[-1]) # [bs*num_win, dim=512] 
                    # print(" > 5. ", task_input.shape, task_input.unsqueeze(0).shape)
                    prob_st = model.task_net(task_input.unsqueeze(0))
                    id_st = torch.zeros(prob_st.shape[0], 1).to(prob_st.device)
                else:
                    raise NotImplementedError

        # task: mt, st(merge:st, mt, at), at need to get output from model.textual_encoder
        if per_task == 'mt' or (per_task == 'at' and sample['mt'] is not None) or multi_st:            # sample_mt --> textual_encoder --> encoder_out_mt
            # print("-- Encode MT Task", per_task)
            sample_mt = sample if per_task != 'at' else sample['mt']   
            if multi_st:
                _, _, prev_output_tokens_mt = sample["net_input"].values()
                src_tokens_mt, src_lengths_mt = sample["transcript"]['tokens'], sample["transcript"]['lengths']
                if self.mixup_for_whole_model and self.mixup_rate != 0.0:
                    assert sample["transcript"]['tokens_add_noise'] is not None
                    assert sample["transcript"]['lengths_add_noise'] is not None
                    src_tokens_mt, src_lengths_mt = sample["transcript"]['tokens_add_noise'], sample["transcript"]['lengths_add_noise']
                
                if update_num < 100 and update_num % 20 == 0:   #  and self.mixup_rate != 0.0
                    print("mix_tag:", self.mix_tag, "at_nopad:", self.at_nopad, "at_low_pos:", self.at_low_pos, "mixup_change_id:", self.mixup_change_id)
                    print("tokens_add_noise:", sample["transcript"]['tokens_add_noise'], "\nlengths_add_noise:", sample["transcript"]['lengths_add_noise'], "\nids:", sample["transcript"]['ids'])
            else: 
                src_tokens_mt, src_lengths_mt, prev_output_tokens_mt = sample_mt["net_input"].values()
            encoder_out_mt = model.textual_encoder(src_tokens_mt, src_lengths_mt)

            # task: mt, st(merge:st, mt, at) need to get output from model.decoder and cal mt loss 
            if per_task == 'mt' or multi_st:     # Cal MT Loss: encoder_out_mt --> decoder --> net_output_mt
                # print("-- Cal MT Loss", per_task)
                net_output_mt = model.decoder(
                    prev_output_tokens=prev_output_tokens_mt, encoder_out=encoder_out_mt
                )

                loss_mt, nll_loss_mt = self.compute_loss(model, net_output_mt, sample_mt, reduce=reduce)
                sample_size_mt = (
                    sample_mt["target"].size(0) if self.sentence_avg else sample_mt["ntokens"]
                )
                logging_output["trans_loss"] += (utils.item(loss_mt.data) if reduce else loss_mt.data)
                logging_output["nll_loss"] += (utils.item(nll_loss_mt.data) if reduce else nll_loss_mt.data)
                logging_output["ntokens"] += sample_mt["ntokens"]
                logging_output["nsentences"] += sample_mt["target"].size(0)
                logging_output["sample_size"] += sample_size_mt

                # print(f"MT Loss {per_task}: {loss_mt}, Sample size {per_task}: {sample_size_mt}")
                sample_size = sample_size_mt if sample_size is None else sample_size + sample_size_mt
                loss = loss_mt if loss is None else loss + loss_mt
            
            # when use self.at_training, task: at, st(merge:st, mt, at) need to get mt output from model.task_net
            if self.at_training and (per_task == 'at' or multi_st):
                # print("-- Record AT data for MT", per_task)
                # required data
                if not self.at_low_pos:
                    encoder_output_mt = encoder_out_mt["encoder_out"][0]                    # [len, bs, dim]
                    encoder_padding_mask_mt = lengths_to_padding_mask(src_lengths_mt) if self.at_nopad else torch.zeros(encoder_out_mt["encoder_padding_mask"][0].shape).bool().to(encoder_out_mt["encoder_padding_mask"][0].device)  # [bs, len]
                else:
                    encoder_output_mt = encoder_out_mt["encoder_embedding"][0]                    # [len, bs, dim]
                    encoder_padding_mask_mt = lengths_to_padding_mask(src_lengths_mt) if self.at_nopad else torch.zeros(encoder_out_mt["encoder_padding_mask"][0].shape).bool().to(encoder_output_mt.device)  # [bs, len]

                if self.at_level == "sentence":
                    if not self.mixup_for_whole_model and self.mixup_rate < 0.0 and sample["transcript"]['tokens_add_noise'] is not None:
                        encoder_output_mt = model.textual_encoder(sample["transcript"]['tokens_add_noise'], sample["transcript"]['lengths_add_noise'])["encoder_out"][0]  
                        encoder_padding_mask_mt = lengths_to_padding_mask(sample["transcript"]['lengths_add_noise']) if self.at_nopad else torch.zeros(encoder_out_mt["encoder_padding_mask"][0].shape).bool().to(encoder_out_mt["encoder_padding_mask"][0].device)  # [bs, len]
                    # dis loss
                    encoder_output_mt_copy = encoder_output_mt.clone().detach_()
                    prob_mt = model.task_net(encoder_output_mt_copy, encoder_padding_mask_mt if self.at_nopad else None)
                    if self.mixup_rate < 0.0 and sample["transcript"]['ids'] is not None and self.mixup_change_id:
                        id_mt = sample["transcript"]['ids'].unsqueeze(-1).to(prob_mt.device)
                    else:
                        id_mt = torch.ones(encoder_padding_mask_mt.shape[0], 1).to(prob_mt.device)
                    # gen loss
                    prob_mt_gen = model.task_net(encoder_output_mt, encoder_padding_mask_mt if self.at_nopad else None)
                    id_mt_gen = torch.full((encoder_padding_mask_mt.shape[0], 1), self.mix_tag).to(prob_mt_gen.device)

                elif self.at_level == "token":
                    encoder_output = encoder_output_mt.transpose(0,1).reshape(-1, encoder_output_mt.shape[-1])   # [bs * len, dim=512]
                    encoder_padding_mask = encoder_padding_mask_mt.reshape(-1)  # [bs * len]
                    encoder_output = encoder_output[~encoder_padding_mask]      # 去除mask为True
                    prob_mt = model.task_net(encoder_output.unsqueeze(0))           # [1, len*bs, dim=512] --> [len*bs, 1]
                    id_mt = torch.ones(prob_mt.shape[0], 1).to(prob_mt.device)                         # [len*bs, 1]
                elif self.at_level == "window":
                    encoder_output = encoder_output_mt.transpose(0,1)   # [bs, len, dim]
                    if self.at_nopad:
                        encoder_output[encoder_padding_mask_mt.unsqueeze(-1).expand(encoder_output.shape)] = 0.0    # 将padding为True的位置设为0，避免影响mean运算
                    tgt_len = sample_mt["transcript"]['tokens'].shape[-1]
                    _shape = encoder_output.shape # [bs, len, dim]
                    length = _shape[1]
                    num_win = max(0, (length - self.at_win['size'])) // self.at_win['stride'] + 1 + (0 if (max(0, (length - self.at_win['size'])) % self.at_win['stride']) == 0 else 1)
                    # print(" > 2. ", length, tgt_len, self.at_win['size'], self.at_win['stride'], num_win)
                    task_input = torch.zeros(num_win, _shape[0], _shape[-1], dtype=torch.float16).to(encoder_output.device)  # [num_win, bs, dim]  to save result
                    # print(" > 3. ", task_input.shape, task_input.dtype)
                    for i in range(num_win):
                        s = i * self.at_win['stride']
                        e = min(s + self.at_win['size'], length)    # 不超过length
                        calc_output = encoder_output[:,s:e,:]   # [bs, size, dim]
                        if self.at_nomute:
                            calc_mask = encoder_padding_mask_mt[:, s:e] # [bs, size]
                            calc_len = (~calc_mask).long().sum(dim=1).unsqueeze(-1) # [bs, 1]
                            calc_len[calc_len==0.0] = 1 # 特殊0值处理
                            task_input[i,:,:] = (calc_output.sum(dim=1) / calc_len)     # [bs, dim=512]
                        else:
                            task_input[i,:,:] = calc_output.mean(dim=1)
                    # print(" > 4. ", task_input.shape)
                    task_input = task_input.transpose(0, 1).reshape(-1, task_input.shape[-1]) # [bs*num_win, dim=512] 
                    # print(" > 5. ", task_input.shape, task_input.unsqueeze(0).shape)
                    prob_mt = model.task_net(task_input.unsqueeze(0))
                    id_mt = torch.ones(prob_mt.shape[0], 1).to(prob_mt.device)         # note kkq change here 08.09.2023
                else:
                    raise NotImplementedError

        # when use self.at_training, task: at, st(merge:st, mt, at) need to cal at loss
        if self.at_training and (per_task == 'at' or multi_st):
            # print("-- Cal AT Loss", per_task)
            # merge st and mt
            task_prob = torch.cat((() if prob_st is None else (prob_st,)) + (() if prob_mt is None else (prob_mt,)), dim=0)
            task_id = torch.cat((() if id_st is None else (id_st,)) + (() if id_mt is None else (id_mt,)), dim=0)
            task_prob_gen = torch.cat((() if prob_st_gen is None else (prob_st_gen,)) + (() if prob_mt_gen is None else (prob_mt_gen,)), dim=0)
            task_id_gen = torch.cat((() if id_st_gen is None else (id_st_gen,)) + (() if id_mt_gen is None else (id_mt_gen,)), dim=0)
            if len(task_prob.shape) == 1:
                print(task_prob, task_prob.shape, task_prob.shape[0])
                assert task_prob.shape[0] == 1
                task_prob = task_prob.unsqueeze(0)
            if len(task_prob_gen.shape) == 1:
                print(task_prob_gen, task_prob_gen.shape, task_prob_gen.shape[0])
                assert task_prob_gen.shape[0] == 1
                task_prob_gen = task_prob_gen.unsqueeze(0)

            # cuda
            # if torch.cuda.is_available():
            #     task_id = task_id.cuda()
            
            sample_size_at = None

            if self.at_level == "sentence":
                weight = torch.cat(
                        (((~encoder_padding_mask_st).long().sum(dim=1),) if ((per_task == 'at' and sample['st'] is not None) or multi_st) else ())
                        + (((~encoder_padding_mask_mt).long().sum(dim=1),) if ((per_task == 'at' and sample['mt'] is not None) or multi_st) else ()),
                        dim=0
                ).unsqueeze(1)                              # [bz] + [bz]  src token length 

                # bce loss dis
                loss_at = F.binary_cross_entropy_with_logits(
                        task_prob.float(),
                        task_id,
                        weight=weight,
                        reduction = "mean"
                )
                # bce loss gen
                loss_at_gen = F.binary_cross_entropy_with_logits(
                        task_prob_gen.float(),
                        task_id_gen,
                        weight=weight,
                        reduction = "mean"
                )
                sample_size_at = task_prob.shape[0]   # [bz + bz]
            elif self.at_level == "token" or self.at_level == 'window':
                mean_set = ((encoder_padding_mask_st.shape[0] if ((per_task == 'at' and sample['st'] is not None) or multi_st) else 0)
                          + (encoder_padding_mask_mt.shape[0] if ((per_task == 'at' and sample['mt'] is not None) or multi_st) else 0))    # bz + bz  =>  num of sentence
                weight = torch.cat(
                            ((torch.ones(prob_st.shape[0], dtype=torch.float16) * (self.at_win['stride'] if self.at_level == "window" else 1),) if ((per_task == 'at' and sample['st'] is not None) or multi_st) else ())
                          + ((torch.ones(prob_mt.shape[0], dtype=torch.float16) * (self.at_win['stride'] if self.at_level == "window" else 1),) if ((per_task == 'at' and sample['mt'] is not None) or multi_st) else ())
                ).unsqueeze(1) / mean_set  # [l, 1]
                # bce loss
                loss_at = F.binary_cross_entropy_with_logits(
                        task_prob.float(),
                        task_id,
                        weight=weight.to(task_prob.device),        # bug
                        reduction = "sum"
                )
                sample_size_at = task_prob.shape[0]               
            else:
                raise NotImplementedError

            # print(encoder_output_st.shape, encoder_output_mt.shape, task_prob.shape)

            if update_num is not None and False:
                # T = 2000.0
                # max_scale = 2 * self.at_scale
                # update_scale =  self.at_scale * (abs(T/2 - (float(update_num) % T)) / (T/2)) * (max_scale / self.at_scale)

                TIME = 20000.0
                END = 45000.0
                update_scale = self.at_scale * min(1.0, (END - min(END, update_num)) / (END - TIME))
                loss_at = loss_at * update_scale       # set scale for at loss
                loss_at_gen = loss_at_gen * update_scale
            else:
                loss_at = loss_at * self.at_scale
                loss_at_gen = loss_at_gen * self.at_scale
                
                mixup_change_id_scale = 1.5
                if self.mixup_change_id:
                    loss_at = loss_at * mixup_change_id_scale
                

            logging_output["task_loss"] = (utils.item(loss_at.data) if reduce else loss_at.data)
            logging_output["task_loss_gen"] = (utils.item(loss_at_gen.data) if reduce else loss_at_gen.data)
            logging_output["sample_size"] += sample_size_at

            # print(f"AT Loss {per_task}: {loss_at}, Sample size {per_task}: {sample_size_at}")
            # merge result
            sample_size = sample_size_at if sample_size is None else sample_size + sample_size_at
            loss = loss_at + loss_at_gen if loss is None else loss + loss_at + loss_at_gen

        # asr task
        if per_task == 'asr':
            # print("-- Cal ASR Loss", per_task)
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
        
        # print(f"TOTAL Loss {per_task}: {loss}, Sample size {per_task}: {sample_size}")
        if multi_st:
            return loss_at, loss_at_gen, loss_st, loss_mt, sample_size, logging_output
        else:
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
                    # with torch.no_grad():
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
                    #with torch.no_grad():
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
        task_loss_gen = utils.item(
            sum(log.get("task_loss_gen", 0) for log in logging_outputs)
        )
        metrics.log_scalar(
            "task_loss_gen",
            task_loss_gen / nsentences / math.log(2),
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
