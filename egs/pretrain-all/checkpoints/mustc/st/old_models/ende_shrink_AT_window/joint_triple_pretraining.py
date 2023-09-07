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

import torch.nn.functional as F


logger = logging.getLogger(__name__)

EVAL_BLEU_ORDER = 4


@register_task("joint_triple_pretraining")
class JointTriplePretrainingTask(SpeechToTextTask):
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
            "--at-level",
            default="sentence",
            type=str,
            help="the level of adversarial training",
        )
        parser.add_argument(
            "--at-nopad",
            default=False,
            type=bool,
            help="rm padding vector when doing at task",
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
        self.at_training = args.adversarial_training
        self.at_level = args.at_level
        self.at_nopad = args.at_nopad
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
        self.asr_weight = 1.0
        self.mt_weight = 1.0
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
            print(state_dict["asr_weight"].cuda())
            self.asr_weight = state_dict["asr_weight"].cuda()
            #self.asr_weight = state_dict["asr_weight"]
        if "mt_weight" in state_dict.keys():
            # pass
            print(state_dict["mt_weight"].cuda())
            self.mt_weight = state_dict["mt_weight"].cuda()
            #self.mt_weight = state_dict["mt_weight"]

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

    def _per_task_train_loss(
        self, per_task, model, update_num, criterion, sample, optimizer, ignore_grad
    ):
        #if per_task == "asr" and update_num > 30000:
        #    ignore_grad=True
        #if per_task == "mt" and update_num > 50000:
        #    ignore_grad=True
        if per_task == "st" and update_num < 3000:        # asronly 不做ctc_shrink去掉                  //need_check//
            ignore_grad=True
            pass

        loss, sample_size, logging_output = criterion(
            model, sample[per_task], per_task
        )
        # print(f"{per_task}: ", sample_size)
        if ignore_grad:
            loss *= 0
        else:
            if per_task == "mt":    
                loss = loss * self.mt_weight   
            elif per_task == "asr":
                loss = loss * self.asr_weight
        optimizer.backward(loss)

        # ************/ Add AT per-task /************ #
        if per_task == 'at':        # reverse the gradients of the encoder
            
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
                tmp_list=[]
                for index in range(12):
                    grad_test1_a=model.acoustic_encoder.w2v_model.encoder.layers[index].self_attn.out_proj.weight
                    #grad_test1_b=model.acoustic_encoder.w2v_model.encoder.layers[index].fc1.weight
                    grad1_a=grad_test1_a.grad.norm()#.cpu().tolist()
                    #grad1_b=grad_test1_b.grad.detach().norm()#.cpu().tolist()
                    tmp_list.append(grad1_a)
                norm_list.append(torch.Tensor(tmp_list).mean().cuda())
                    #print("acoustic layer",grad1_a,grad1_b)
            if per_task != "asr":
                tmp_list=[]
                for index in range(6):
                    grad_test2_a=model.acoustic_encoder.sead.layers[index].self_attn.out_proj.weight
                    #grad_test2_b=model.acoustic_encoder.sead.layers[index].fc1.weight
                    grad2_a=grad_test2_a.grad.norm()#.cpu().tolist()
                    #grad2_b=grad_test2_b.grad.detach().norm()#.cpu().tolist()
                    tmp_list.append(grad2_a)
                norm_list.append(torch.Tensor(tmp_list).mean().cuda())
                tmp_list=[]
                for index in range(6):
                    grad_test2_d=model.decoder.layers[index].self_attn.out_proj.weight
                    grad2_d=grad_test2_d.grad.norm()
                    tmp_list.append(grad2_d)
                norm_list.append(torch.Tensor(tmp_list).mean().cuda())
                    #print("encoder layer", grad2_a, grad2_b)
                    #with open("textual_encoder_norm","a") as write_file:
                    #    np.savetxt(write_file, grad2, fmt="%.8f",newline=" ",footer="\n",comments="")
                    #    write_file.write(str(grad2)+"\n")

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
                
        # ************/ Add AT per-task /************ #
        if ('st' in sample.keys() or 'mt' in sample.keys()) and self.at_training is True:
            sample["at"] = {'st':sample['st'] if 'st' in sample.keys() else None, 'mt':sample['mt'] if 'mt' in sample.keys() else None} # add at sample
            sample.move_to_end('at', False) # change at task first
        # print(f"per task list:{sample.keys()}")

        for idx, per_task in enumerate(sample.keys()): #enumerate(["st","asr","mt"]):
            if per_task not in sample.keys():
                continue
            else:
                if per_task == "asr" and (update_num > 30000 or self.asr_weight < 0.1): # 30000 # 100000 asronly            //need_check//
                    continue
                if per_task == "mt" and (update_num > 50000 or self.mt_weight < 0.1): # 50000 # 100000 ende_shrink_AT_mt    //need_check//
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
            if self.asr_weight >= 0.1 and update_num < 30000 :
                if asr_list and st_list:
                    tmp_asr_weight = self.asr_weight * (asr_list[0] / st_list[0]) ** coe
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
                if self.asr_weight >= 0.1 and update_num < 30000:
                    #handle = dist.all_reduce(output, async_op=True)
                    #handle.wait()
                    dist.all_reduce(tmp_asr_weight)
                    self.asr_weight = tmp_asr_weight / world_size 
                if self.mt_weight >= 0.1 and update_num < 50000:
                    dist.all_reduce(tmp_mt_weight)
                    self.mt_weight = tmp_mt_weight / world_size       # ende_shrink_AT_mt 不更新mt权重：去掉这句                          //need_check//
            self.state.merge_state_dict({"asr_weight": self.asr_weight})
            self.state.merge_state_dict({"mt_weight": self.mt_weight})
            print(self.mt_weight) #(mt_list[0] / st_list[1]), (mt_list[1] / st_list[2]))
            print(self.asr_weight) #(asr_list[0] / st_list[0]))
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
            if 'st' in sample.keys()  and self.at_training is True:
                # sample["at"] = {'st':sample['st'] if 'st' in sample.keys() else None, 'mt':sample['mt'] if 'mt' in sample.keys() else None} # add at sample
                sample["at"] = {'st':sample['st'], 'mt': None}
                
            for idx, per_task in enumerate((self.all_tasks + ['at']) if ('st' in sample.keys() and self.at_training is True) else self.all_tasks):
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
    
    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):

        with torch.no_grad():
            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, constraints=constraints
            )

        #return SpeechToTextDataset(
        #    "interactive", False, self.data_cfg, src_tokens, src_lengths
        #)
