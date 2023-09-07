# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os.path as op
import numpy as np
from argparse import Namespace
from collections import OrderedDict
import contextlib
import torch
import json
from typing import Any, Callable, Dict, List

from fairseq import metrics, utils
from fairseq.data import Dictionary, encoders, RoundRobinZipDatasets
from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDataset,
    SpeechToTextDatasetCreator,
    get_features_or_waveform
)
from fairseq.data.audio.triple_dataset import TripleDatasetCreator,S2TTripleDataConfig,TripleDataset

from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.tasks.speech_to_text import SpeechToTextTask
import torch.distributed as dist

from fairseq.data.data_utils import lengths_to_padding_mask

import torch.nn.functional as F


logger = logging.getLogger(__name__)

EVAL_BLEU_ORDER = 4


@register_task("joint_triple_pretraining_merge")
class JointTriplePretrainingMergeTask(SpeechToTextTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        # ************/ Add AT per-task /************ #
        parser.add_argument(
            "--adversarial-training",
            default=False,
            type=bool,
            help="if use adversarial training",
        )
        parser.add_argument(
            "--mixup-rate",
            default=0.0,
            type=float,
            help="the rat of mixup in acoustic_encoder",
        )
        parser.add_argument(
            "--at-level",
            default="sentence",
            type=str,
            help="the level of adversarial training",
        )
        parser.add_argument(
            "--at-scale",
            default=1,
            type=float,
            help="the scale of at loss",
        )
        parser.add_argument(
            "--at-adapte-win",
            default=False,
            type=bool,
            help="reset window size and stride by calculating src length and tgt length",
        )
        parser.add_argument(
            "--at-nopad",
            default=False,
            type=bool,
            help="rm padding vector for text when doing at task",
        )
        parser.add_argument(
            "--at-nomute",
            default=False,
            type=bool,
            help="rm mute vector for speech when doing at task",
        )
        parser.add_argument(
            "--keep-mt-task",
            default=False,
            type=bool,
            help="keep doing mt task and do not update mt weight",
        )
        parser.add_argument(
            "--merge-mt-st",
            default=False,
            type=bool,
            help="merge st task and mt task to speech up training process, works only when keep-mt-task is True",
        )
        parser.add_argument(
            "--embedding-l2norm",
            default=False,
            type=bool,
            help="use L2 Norm",
        )
        parser.add_argument(
            "--max-source-positions",
            default=6000,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )
        parser.add_argument(
            "--weight-steps",
            default=5000,
            type=int,
            metavar="N",
            help="number of per step to update task weight",
        )
        parser.add_argument(
            "--eval-bleu", 
            action="store_true",
            help= "evaluation with BLEU scores",
        )
        parser.add_argument(
            "--eval-bleu-args",
            type=str,
            default="{}",
            help="generation args for BLUE scoring, as JSON string",
        )
        parser.add_argument(
            "--eval-bleu-detok", 
            type=str,
            default="space",
            help= "detokenize before computing BLEU (e.g., 'moses'); required if using --eval-bleu; use 'space' to disable detokenization; see fairseq.data.encoders for other options"
        )
        parser.add_argument(
            "--eval-bleu-detok-args",
            type=str,
            default="{}",
            help= "args for building the tokenizer, if needed, as JSON string",
        )
        parser.add_argument(
            "--eval-tokenized-bleu",
            action="store_true",
            help= "compute tokenized BLEU instead of sacrebleu"
        )
        parser.add_argument(
            "--eval-bleu-remove-bpe",
            type=str,
            default=None,
            help= "remove BPE before computing BLEU argparse_const",
        )

    def __init__(self, args, tgt_dict, src_dict=None, cluster_convert=None):
        super().__init__(args, tgt_dict)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.cluster_convert = cluster_convert
        self.all_tasks = ["st","mt","asr"]
        # ************/ Add AT per-task /************ #
        self.args = args
        self.at_training = args.adversarial_training
        self.at_level = args.at_level
        self.at_scale = args.at_scale 
        self.at_adapte_win = args.at_adapte_win
        self.at_nopad = args.at_nopad
        self.at_nomute = args.at_nomute
        self.mixup_rate = args.mixup_rate
        # ************/ Add Task set /************ #
        self.keep_mt_task = args.keep_mt_task
        self.merge_mt_st = args.merge_mt_st
        if self.merge_mt_st:
            assert self.keep_mt_task, "merge-mt-st is supported only when keep_mt_task is TRUE"
        # ************/ Add L2 norm /************ #
        self.embedding_l2norm = args.embedding_l2norm
        
        self.data_cfg = S2TTripleDataConfig(op.join(args.data, args.config_yaml))
        self.eval_bleu = args.eval_bleu
        self.eval_bleu_remove_bpe = args.eval_bleu_remove_bpe
        self.eval_tokenized_bleu = args.eval_tokenized_bleu
        self.eval_bleu_remove_bpe = args.eval_bleu_remove_bpe
        self.weight_steps = args.weight_steps
        self.state.add_factory("asr_weight", self.load_asr_weight)
        self.state.add_factory("mt_weight", self.load_mt_weight)
        self.asr_weight = torch.tensor(1.0)
        self.mt_weight = torch.tensor(1.0)          # //need_check//              
        logger.info(
                f"Initial task weight: asr {self.state.asr_weight}: " f"mt {self.state.mt_weight}"
            )

    def load_asr_weight(self):
        return self.asr_weight

    def load_mt_weight(self):
        return self.mt_weight

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = S2TTripleDataConfig(op.join(args.data, args.config_yaml))
        dict_path = op.join(args.data, data_cfg.vocab_filename)
        if not op.isfile(dict_path):
            raise FileNotFoundError(f"Dict not found: {dict_path}")
        tgt_dict = Dictionary.load(dict_path)
        logger.info(
            f"dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}"
        )

        src_dict = None
        if getattr(data_cfg, "share_src_and_tgt", False):
            asr_vocab_filename = data_cfg.vocab_filename
        else:
            asr_vocab_filename = getattr(data_cfg, "asr_vocab_filename", None)
        if asr_vocab_filename is not None:
            dict_path = op.join(args.data, asr_vocab_filename)
            if not op.isfile(dict_path):
                raise FileNotFoundError(f"Dict not found: {dict_path}")
            src_dict = Dictionary.load(dict_path)
            logger.info(
                f"asr dictionary size ({asr_vocab_filename}): " f"{len(src_dict):,}"
            )

        cluster_dict = getattr(data_cfg, "cluster_dict", None)
        if cluster_dict is not None:
            dict_path = op.join(args.data, cluster_dict)
            if not op.isfile(dict_path):
                raise FileNotFoundError(f"Dict not found: {dict_path}")
            cluster_convert = {}
            with open(dict_path,"r") as f:
                lines=f.read().strip().split("\n")
                for line in lines:
                    cluster_convert[int(line.split(" ")[1])] = int(line.split(" ")[2])
            logger.info(
                f"cluster dictionary size ({cluster_dict}): " f"{len(cluster_convert):,}"
            )

        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')
        return cls(args, tgt_dict, src_dict, cluster_convert)

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if "asr_weight" in state_dict.keys():
            # pass 
            print("asr_weight", state_dict["asr_weight"])
            #self.asr_weight = torch.tensor(state_dict["asr_weight"]).cuda()
            self.asr_weight = state_dict["asr_weight"]
        if "mt_weight" in state_dict.keys():
            # pass
            print("mt_weight", state_dict["mt_weight"])
            #self.mt_weight = torch.tensor(state_dict["mt_weight"]).cuda()
            self.mt_weight = state_dict["mt_weight"]

    def build_criterion(self, args):
        from fairseq import criterions

        if self.data_cfg.prepend_tgt_lang_tag and args.ignore_prefix_size != 1:
            raise ValueError(
                'Please set "--ignore-prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        return criterions.build_criterion(args, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        is_valid_split = split.startswith("dev") or split.startswith("valid")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        if self.data_cfg.src_bpe_tokenizer is not None:
            src_bpe_tokenizer = self.build_src_bpe(self.args)
        else:
            src_bpe_tokenizer = bpe_tokenizer
            # if self.data_cfg.share_src_and_tgt:
            #     src_bpe_tokenizer = bpe_tokenizer
            # else:
            #     src_bpe_tokenizer = None
        is_decode=True
        if is_train_split:
            is_decode=False
            train_files = split.split(",")
            st_files = []
            mt_files = []
            asr_files = [] 
            for file_name in train_files:
                if "st" in file_name:
                    st_files.append(file_name)
                elif "mt" in file_name:
                    mt_files.append(file_name)
                elif "asr" in file_name:
                    asr_files.append(file_name)
                else:
                    raise ValueError(
                    'Please specify the file type, the file name should contain one of "st,mt,asr" tag'
                    )
            split_st = ','.join(st_files)
            split_mt = ','.join(mt_files)
            split_asr = ','.join(asr_files)

            data_dict=[]


            if len(mt_files) > 0:
                mt_data = TripleDatasetCreator.from_tsv(
                    self.args.data,
                    self.data_cfg,
                    split_mt,
                    self.tgt_dict,
                    pre_tokenizer,
                    bpe_tokenizer,
                    is_train_split=is_train_split,
                    epoch=epoch,
                    seed=self.args.seed,
                    src_dict=self.src_dict,
                    src_bpe_tokenizer=src_bpe_tokenizer,
                    data_type="mt"
                )
                data_dict.append(("mt",mt_data))


            if len(asr_files) > 0:
                asr_data = TripleDatasetCreator.from_tsv(
                    self.args.data,
                    self.data_cfg,
                    split_asr,
                    self.tgt_dict,
                    pre_tokenizer,
                    bpe_tokenizer,
                    is_train_split=is_train_split,
                    epoch=epoch,
                    seed=self.args.seed,
                    src_dict=self.src_dict,
                    src_bpe_tokenizer=src_bpe_tokenizer,
                    cluster_convert=self.cluster_convert,
                    data_type="asr"
                )
                data_dict.append(("asr",asr_data))

            if len(st_files) > 0:
                st_data = TripleDatasetCreator.from_tsv(
                    self.args.data,
                    self.data_cfg,
                    split_st,
                    self.tgt_dict,
                    pre_tokenizer,
                    bpe_tokenizer,
                    is_train_split=is_train_split,
                    epoch=epoch,
                    seed=self.args.seed,
                    src_dict=self.src_dict,
                    src_bpe_tokenizer=src_bpe_tokenizer,
                    cluster_convert=self.cluster_convert,
                    data_type="st"
                )
                data_dict.append(("st",st_data))

            self.datasets[split] = RoundRobinZipDatasets(
                OrderedDict(
                        data_dict
                )
            )
        else:
            if split.endswith("_st") or split.endswith("_mt") or split.endswith("_asr"):
                task_name = split.split("_")[-1]
                st_infer_data= TripleDatasetCreator.from_tsv(
                    self.args.data,
                    self.data_cfg,
                    split,
                    self.tgt_dict,
                    pre_tokenizer,
                    bpe_tokenizer,
                    is_train_split=is_train_split,
                    epoch=epoch,
                    seed=self.args.seed,
                    src_dict=self.src_dict,
                    src_bpe_tokenizer=src_bpe_tokenizer,
                    cluster_convert=self.cluster_convert,
                    data_type=task_name
                )
                self.datasets[split] = RoundRobinZipDatasets(
                    OrderedDict(
                        [
                            (task_name, st_infer_data)
                        ]
                    ),
                    eval_key=None
                    if is_train_split or is_valid_split
                    else task_name,
                )
            else:
                raise NotImplementedError("Do not support decoding this task.")

    def build_model(self, args):
        args.input_feat_per_channel = self.data_cfg.input_feat_per_channel
        args.input_channels = self.data_cfg.input_channels
        models = super(SpeechToTextTask, self).build_model(args)
        if self.eval_bleu:
            detok_args = json.loads(args.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=args.eval_bleu_detok, **detok_args)
            )

            gen_args = json.loads(args.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [models], Namespace(**gen_args)
            )

        return models

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        if self.data_cfg.prepend_tgt_lang_tag and args.prefix_size != 1:
            raise ValueError(
                'Please set "--prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        lang_token_ids = {
            i
            for s, i in self.tgt_dict.indices.items()
            if TripleDataset.is_lang_tag(s)
        }
        extra_gen_cls_kwargs = {"symbols_to_strip_from_output": lang_token_ids}
        return super().build_generator(
            models, args, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def cal_w2v_model_grad(self, model):
        tmp_list=[]
        for index in range(12):
            grad_test1_a=model.acoustic_encoder.w2v_model.encoder.layers[index].self_attn.out_proj.weight
            #grad_test1_b=model.acoustic_encoder.w2v_model.encoder.layers[index].fc1.weight
            grad1_a=grad_test1_a.grad.norm()#.cpu().tolist()
            #grad1_b=grad_test1_b.grad.detach().norm()#.cpu().tolist()
            tmp_list.append(grad1_a)
        return tmp_list

    def cal_textual_encoder_decoder_grad(self, model):
        encoder_tmp_list=[]
        for index in range(6):
            grad_test2_a=model.acoustic_encoder.sead.layers[index].self_attn.out_proj.weight
            #grad_test2_b=model.acoustic_encoder.sead.layers[index].fc1.weight
            grad2_a=grad_test2_a.grad.norm()#.cpu().tolist()
            #grad2_b=grad_test2_b.grad.detach().norm()#.cpu().tolist()
            encoder_tmp_list.append(grad2_a)
        decoder_tmp_list=[]
        for index in range(6):
            grad_test2_d=model.decoder.layers[index].self_attn.out_proj.weight
            grad2_d=grad_test2_d.grad.norm()
            decoder_tmp_list.append(grad2_d)
        return encoder_tmp_list, decoder_tmp_list

    def _per_task_train_loss(
        self, per_task, model, update_num, criterion, sample, optimizer, ignore_grad
    ):
        #if per_task == "asr" and update_num > 30000:
        #    ignore_grad=True
        #if per_task == "mt" and update_num > 50000:
        #    ignore_grad=True
        if per_task == "st" and update_num < 3000:        # asronly 不做ctc_shrink去掉                  //need_check//
            ignore_grad=True
        # print(f"Doing {per_task} Task")
        # ************/ Merge ST&MT&AT Task /************ #
        if self.merge_mt_st and per_task == 'st':       # Task合并的ST 这里有(AT,)MT,ST(三)两项任务
            
            loss_at, loss_st, loss_mt, sample_size, logging_output = criterion(
                model, sample[per_task], per_task, merge_task=True, st_update=(update_num > 3000), update_num=update_num
            )
            # print(f"at:{loss_at}\t st:{loss_st}\t mt:{loss_mt}")
            # torch.autograd.set_detect_anomaly(True)
            
            if ignore_grad:     # 前3k步不更新st任务
                loss_st = loss_st * 0
            
            if self.at_training: #  and not ignore_grad:
                # print(f"--Backword AT Loss", loss_at)
                # optimizer.backward(loss_at)
                loss_at.backward(retain_graph=True)                
                # reverse the gradients of the encoders           
                for name, param in model.acoustic_encoder.named_parameters():
                    if param.grad is not None:
                        param.grad = -1 * param.grad
                for name, param in model.textual_encoder.layer_norm.named_parameters():
                    if param.grad is not None:
                        param.grad = -1 * param.grad
            
            loss_mt_st = loss_st + loss_mt
            # print(f"--Backword ST+MT Loss", loss_mt_st)
            optimizer.backward(loss_mt_st)
            
            norm_list = []
            if update_num % self.weight_steps == 0 and update_num > 3000 and per_task == 'st':  # 合并的情况下只需考虑st任务的grad提供给计算asr_weight
                w2v_tmp_list = self.cal_w2v_model_grad(model)
                encoder_tmp_list, decoder_tmp_list = self.cal_textual_encoder_decoder_grad(model)
                norm_list.append(torch.Tensor(w2v_tmp_list).mean().cuda())
                norm_list.append(torch.Tensor(encoder_tmp_list).mean().cuda())
                norm_list.append(torch.Tensor(decoder_tmp_list).mean().cuda())
            
            # optimizer.backward(loss_mt)
            # 暂时不需要对MT任务的grad进行计算，因为无需更新mt_weight，若要更新，需要对_per_task_train_loss函数的return进行修改
            assert self.keep_mt_task, "BUG HERE: When using MERGE_MT_ST, args KEEP_MT_TASK should be True!"
            
            loss = (loss_at if loss_at else 0) + (loss_st if loss_st else 0) + (loss_mt if loss_mt else 0)
        
        else:                                           # Task不合并 或者 合并的ASR任务
            
            loss, sample_size, logging_output = criterion(
                model, sample[per_task], per_task
            )
            if ignore_grad:
                loss *= 0
            else:
                if per_task == "mt":    
                    loss = loss * self.mt_weight   
                elif per_task == "asr":
                    loss = loss * self.asr_weight
            
            optimizer.backward(loss)

            # ************/ Add AT per-task /************ #
            if per_task == 'at':        # reverse the gradients of the encoders
                # here are shared params between two encoders            
                # acoustic encoder
                for name, param in model.acoustic_encoder.named_parameters():
                    if param.grad is not None:
                        param.grad = -1 * param.grad
                # textual encoder
                for name, param in model.textual_encoder.layer_norm.named_parameters():
                    if param.grad is not None:
                        param.grad = -1 * param.grad
            
            norm_list = []
            if update_num % self.weight_steps == 0 and update_num > 3000 and per_task != 'at':  # skip at task
                if per_task != "mt":
                    tmp_list = self.cal_w2v_model_grad(model)
                    norm_list.append(torch.Tensor(tmp_list).mean().cuda())
                if per_task != "asr":
                    encoder_tmp_list, decoder_tmp_list = self.cal_textual_encoder_decoder_grad(model)
                    norm_list.append(torch.Tensor(encoder_tmp_list).mean().cuda())
                    norm_list.append(torch.Tensor(decoder_tmp_list).mean().cuda())

        return loss, sample_size, logging_output, norm_list

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)
        from collections import defaultdict

        agg_loss, agg_sample_size, agg_logging_output = 0.0, 0.0, defaultdict(float)

        # TODO the data in smaple and all_task is not match, the sample may lack of some types data
        #mt_grad_a_list = []
        #mt_grad_b_list = []
        st_list = None
        asr_list = None
        mt_list = None

        # ************/ Add AT per-task (No Merge) /************ #
        if ('st' in sample.keys() or 'mt' in sample.keys()) and self.at_training and not self.merge_mt_st:              # 不进行合并时，单独添加AT任务
            sample["at"] = {'st':sample['st'] if 'st' in sample.keys() else None, 'mt':sample['mt'] if 'mt' in sample.keys() else None}
            sample.move_to_end('at', False) # change at task first
        
        # ************/ Merge ST&MT&AT Task /************ # 
        task_list = list(sample.keys())      
        if self.merge_mt_st:                        # 合并的话去掉MT的索引
            task_list = ['st', 'asr']

        for idx, per_task in enumerate(task_list): #enumerate(["st","asr","mt"]):
            if per_task not in sample.keys():
                continue
            else:
                if per_task == "asr" and (update_num > 15000 or self.asr_weight < 0.1):
                    continue
                if per_task == "mt" and (update_num > 50000 or self.mt_weight < 0.1) and not self.keep_mt_task:
                    continue
            #if per_task == "st" and update_num < 3000:
            #    continue
            def maybe_no_sync():
                if (
                    self.args.distributed_world_size > 1
                    and hasattr(model, "no_sync")
                    and idx < len(self.all_tasks) - 1
                ):
                    return model.no_sync()
                else:
                    return contextlib.ExitStack()  # dummy contextmanager

            with maybe_no_sync():
                loss, sample_size, logging_output, norm_list = self._per_task_train_loss(
                    per_task,
                    model,
                    update_num,
                    criterion,
                    sample,
                    optimizer,
                    ignore_grad,
                )
            
            if per_task == "st":
                st_list = norm_list
            elif per_task == "asr":
                asr_list = norm_list
            elif per_task == "mt":
                mt_list = norm_list

            agg_loss += loss.detach().item()
            # TODO make summing of the sample sizes configurable
            agg_sample_size += sample_size
            for k in logging_output:
                agg_logging_output[k] += logging_output[k]
                #agg_logging_output[f"{task_pair}:{k}"] += logging_output[k]
                #print(k)
        
        if update_num % self.weight_steps == 0 and update_num != 0 :
            coe = int(update_num / self.weight_steps) 
            if self.asr_weight >= 0.1 and update_num < 15000 :
                if asr_list and st_list:
                    # tmp_asr_weight = self.asr_weight * (asr_list[0] / st_list[0]) ** coe
                    tmp_asr_weight = self.asr_weight * (st_list[0] / asr_list[0]) ** coe
                else:
                    tmp_asr_weight = self.asr_weight 
                #if self.asr_weight - tmp_asr_weight  < 0.01 : 
                #    self.asr_weight = torch.Tensor([0]).cuda()
                #else:
            if self.mt_weight >= 0.1 and update_num < 50000:
                if mt_list and st_list:
                    tmp_mt_weight = self.mt_weight * max(mt_list[0] / st_list[1], mt_list[1] / st_list[2]) ** (coe / 2)
                    #tmp_mt_weight = self.mt_weight * (((mt_list[0] / st_list[1]) + (mt_list[1] / st_list[2])) / 2) ** (coe / 2)
                else:
                    tmp_mt_weight = self.mt_weight 
                #if self.mt_weight - tmp_mt_weight < 0.01 : 
                #    self.mt_weight = torch.Tensor([0]).cuda()
                #else:
                #print("mt_wight",tmp_mt_weight)
            if torch.distributed.is_initialized():
                world_size = dist.get_world_size()
                if self.asr_weight >= 0.1 and update_num < 15000:
                    #handle = dist.all_reduce(output, async_op=True)
                    #handle.wait()
                    dist.all_reduce(tmp_asr_weight)
                    self.asr_weight = tmp_asr_weight / world_size 
                if self.mt_weight >= 0.1 and update_num < 50000 and not self.keep_mt_task:
                    dist.all_reduce(tmp_mt_weight)
                    self.mt_weight = tmp_mt_weight / world_size
            self.state.merge_state_dict({"asr_weight": self.asr_weight})
            self.state.merge_state_dict({"mt_weight": self.mt_weight})
            print("mt_weight", self.mt_weight) # (mt_list[0] / st_list[1]), (mt_list[1] / st_list[2]))
            print("asr_weight", self.asr_weight) # (asr_list[0] / st_list[0]))
        return agg_loss, agg_sample_size, agg_logging_output

    def _per_task_pair_valid_loss(self, per_task, model, criterion, sample):
        loss, sample_size, logging_output = criterion(model, sample[per_task], per_task) 
        
        # ************/ Add AT per-task /************ #
        if per_task == 'at':
            return loss, sample_size, logging_output
        
        if self.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample[per_task], model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.eval_bleu:

            def sum_logs(key):
                import torch

                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect

                    try:
                        from sacrebleu.metrics import BLEU

                        comp_bleu = BLEU.compute_bleu
                    except ImportError:
                        # compatibility API for sacrebleu 1.x
                        import sacrebleu

                        comp_bleu = sacrebleu.compute_bleu

                    fn_sig = inspect.getfullargspec(comp_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = comp_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=int(meters["_bleu_sys_len"].sum),
                        ref_len=int(meters["_bleu_ref_len"].sum),
                        **smooth,
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            from collections import defaultdict

            agg_loss, agg_sample_size, agg_logging_output = 0.0, 0.0, defaultdict(float)
            
            # ************/ Add AT per-task /************ #
            if 'st' in sample.keys() and self.at_training:
                # sample["at"] = {'st':sample['st'] if 'st' in sample.keys() else None, 'mt':sample['mt'] if 'mt' in sample.keys() else None} # add at sample
                sample["at"] = {'st':sample['st'], 'mt': None}
            
            for idx, per_task in enumerate((self.all_tasks + ['at']) if ('st' in sample.keys() and self.at_training) else self.all_tasks):
                if (
                    per_task not in sample
                    or sample[per_task] is None
                    or len(sample[per_task]) == 0
                ):
                    continue
                loss, sample_size, logging_output = self._per_task_pair_valid_loss(per_task, model, criterion, sample)
                agg_loss += loss.data.item()
                # TODO make summing of the sample sizes configurable
                agg_sample_size += sample_size
                for k in logging_output:
                    agg_logging_output[k] += logging_output[k]
                    agg_logging_output[f"{per_task}:{k}"] += logging_output[k]
            #agg_loss, agg_sample_size, agg_logging_output = criterion(model, sample, "st")
        return agg_loss, agg_sample_size, agg_logging_output

    def build_src_bpe(self, args):
        logger.info(f"src tokenizer: {self.data_cfg.src_bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.src_bpe_tokenizer))

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = [get_features_or_waveform(p).shape[0] for p in lines]
        return lines, n_frames

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        #TODO only for speech translation task
        st_infer_data=TripleDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )
        return RoundRobinZipDatasets(
            OrderedDict(
                [
                    ("st", st_infer_data)
                ]
            )
        )
    
    def save_top_output(self, models, sample):

        # set pic name
        # exp_name = "ende_shrink_AT_mt"
        exp_name = self.args.results_path.split('/')[-1]
        level = 'sentence'
         
        current_directory = '/mnt/zhangyh/fairseq-AT/egs/pretrain-all'
        temp_dir = current_directory + '/pic/temp'
        # print(exp_name, temp_dir)
        # assert False
        
        # model input
        st_src_tokens, st_src_lengths = sample["net_input"]['src_tokens'], sample["net_input"]['src_lengths']
        mt_src_tokens, mt_src_lengths = sample["transcript"]['tokens'], sample["transcript"]['lengths']
        # print(st_src_tokens.shape, st_src_lengths.shape, mt_src_tokens.shape, mt_src_lengths.shape)
        st_encoder_out = models[0].acoustic_encoder(st_src_tokens, st_src_lengths)
        mt_encoder_out = models[0].textual_encoder(mt_src_tokens, mt_src_lengths)
        # model output
        st_embedding = st_encoder_out["wav2vec_feature"][0].transpose(0,1).cpu()
        st_encoder_output = st_encoder_out["encoder_out"][0].transpose(0,1).cpu()                   # [bs, len, dim]
        mt_embedding = mt_encoder_out["encoder_embedding"][0].cpu()
        mt_encoder_output = mt_encoder_out["encoder_out"][0].transpose(0,1).cpu()                    # [bs, len, dim]
        
        # assert False
        # preprocess
        if level == 'sentence':
            st_embedding = st_embedding.mean(dim=1)   # [bs, dim]
            st_encoder_output = st_encoder_output.mean(dim=1)   # [bs, dim]
            
            mt_embedding = mt_embedding.mean(dim=1)   # [bs, dim]
            mt_encoder_output = mt_encoder_output.mean(dim=1)   # [bs, dim]

        # save
        spe_temp_file = temp_dir + '/' + exp_name + '_' + level + '_speech.pt'
        txt_temp_file = temp_dir + '/' + exp_name + '_' + level + '_text.pt'

        if op.isfile(spe_temp_file):
            data = torch.load(spe_temp_file)
            st_embedding = torch.cat((st_embedding, data['input']), dim=0)
            st_encoder_output = torch.cat((st_encoder_output, data['output']), dim=0)
            
        torch.save({'output':st_encoder_output, 'input':st_embedding}, spe_temp_file)

        if op.isfile(txt_temp_file):
            data = torch.load(txt_temp_file)
            mt_embedding = torch.cat((mt_embedding, data['input']), dim=0)
            mt_encoder_output = torch.cat((mt_encoder_output, data['output']), dim=0)

        torch.save({'output':mt_encoder_output, 'input':mt_embedding}, txt_temp_file)

        # print("Name:\t{}\nSave:\t{}\t{}".format(exp_name, spe_temp_file, txt_temp_file))
        # assert False

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        if True and (sample['transcript']['tokens'] is not None) and ('results_path' in dir(self.args.results_path)):       # //need_check//        
            self.save_top_output(models, sample)

        with torch.no_grad():
            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, constraints=constraints
            )

        #return SpeechToTextDataset(
        #    "interactive", False, self.data_cfg, src_tokens, src_lengths
        #)
