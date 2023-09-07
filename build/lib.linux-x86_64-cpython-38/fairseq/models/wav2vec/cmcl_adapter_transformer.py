# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import contextlib
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from omegaconf import MISSING, II, open_dict
from typing import Any, Optional
import logging

from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.tasks import FairseqTask
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
)
from fairseq.models.transformer import TransformerEncoder
from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
from fairseq.models.wav2vec.cmcl_adapter import Conv1dSubsampler 
from fairseq.modules import LayerNorm, PositionalEmbedding, TransformerDecoderLayer, TransformerEncoderLayer, ConformerEncoderLayer,RelPositionalEncoding, ConvolutionModule


@dataclass
class CmclAdapterTransformerConfig(FairseqDataclass):
    w2v_path: str = field(
        default=MISSING, metadata={"help": "path to wav2vec 2.0 model"}
    )
    mt_model_path: str = field(
        default=MISSING, metadata={"help": "path to pretrained mt model"}
    )
    mt_model_filter_size: int = field(
        default=0, metadata={"help": "ffn filter size of model"}
    )
    embed_path: str = field(
        default="", metadata={"help": "path to word embed"}
    )
    embed_dim: int = field(
        default=1024, metadata={"help": "mt model word embed size"}
    )
    tune_w2v_LNA: bool = field(
        default=False, metadata={"help": "Tune wav2vec layer norm and self attn"}
    )
    no_pretrained_weights: bool = field(
        default=False, metadata={"help": "if true, does not load pretrained weights"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )
    dropout: float = field(
        default=0.0, metadata={"help": "dropout probability inside wav2vec 2.0 model"}
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside wav2vec 2.0 model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside wav2vec 2.0 model"
        },
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask (normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )

    # channel masking
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )
    freeze_finetune_updates: int = field(
        default=0, metadata={"help": "dont finetune wav2vec for this many updates"}
    )
    feature_grad_mult: float = field(
        default=0.0, metadata={"help": "reset feature grad mult in wav2vec 2.0 to this"}
    )
    layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a layer in wav2vec 2.0"}
    )
    normalize: bool = II("task.normalize")
    conv_kernel_sizes: str = field(
        default="3",
        metadata={"help": "kernel sizes of Conv1d subsampling layers"}
    )

    conv_channels: int = field(
        default=1024,
        metadata={"help": "size of channels in Conv1d subsampling layers"}
    )

    use_cnn_module: bool = field(
        default=False,
        metadata={"help":"Use convolution module or not"}
    )
    adapter_layers: int = field(
        default=0,
        metadata={"help":"conformer layers of normal adapter"}
    )
    sead_layers: int = field(
        default=0,
        metadata={"help":"conformer layers of semantic adapter"}
    )
    contrastive_mask_prob: float = field(
        default=0.0,
        metadata={"help":"drop some feature to boost contrastive learning"}
    )
    encoder_embed_dim: int = field(
        default=1024,
        metadata={"help":"conformer layer size of adapter"}
    )
    #adapter conformer
    macaron_style: bool = field(
        default=False,
        metadata={"help":"use macaron style or not"}
    )
    relative_attn: bool = field(
        default=False,
        metadata={"help":"use relative position attention"}
    )
    cnn_module_kernel: int = field(
        default=31,
        metadata={"help":"cnn module kernel size"}
    )
    encoder_ffn_embed_dim: int = field(
        default=4096,
        metadata={"help":"adapter ffn size"}
    )
    encoder_attention_heads: int = field(
        default=16,
        metadata={"help":"adapter head size"}
    )
    encoder_normalize_before: bool = field(
        default=True,
        metadata={"help":"pre-normalize or post-normalize"}
    )
    position_unit_size: int = field(
        default=0,
        metadata={"help":"the unit size to predict the speech position"}
    )
    add_position_embed: bool = field(
        default=True,
        metadata={"help":"add position embed"}
    )
    add_position_embed_after_ctc: bool = field(
        default=False,
        metadata={"help":"add position embed after ctc loss"}
    )
    freeze_modules: str =field(
        default="",
        metadata={"help":"freezing module"}
    )
    use_ctc_loss: bool =field(
        default=False,
        metadata={"help":"using ctc loss"}
    ) 
    local_rank: int =field(
        default=0,
        metadata={"help":"distribute training"}
    )
    data: str = II("task.data")
    
    # this holds the loaded wav2vec args
    w2v_args: Any = None
    mt_model_args: Any = None




@register_model("cmcl_adapter_transformer", dataclass=CmclAdapterTransformerConfig)
class CmclAdapterTransformer(BaseFairseqModel):
    def __init__(self, cfg: CmclAdapterTransformerConfig, w2v_encoder: BaseFairseqModel, mt_model_encoder: BaseFairseqModel):
        super().__init__()
        #import argparse
        #parser = argparse.ArgumentParser()
        #parser.add_argument("--local_rank", type=int)
        #args = parser.parse_args()
        #cfg.local_rank=args.local_rank
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder
        self.mt_model_encoder = mt_model_encoder

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: CmclAdapterTransformerConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = Wav2vecEncoder(cfg, task.target_dictionary)

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            if path and path != "":
                logging.info("load pretrain embeddings: {}".format(path))
                embed_dict = utils.parse_embedding(path)
                pretrain_embedding_dim=list(list(embed_dict.values())[0].size())[0]
                if pretrain_embedding_dim != embed_dim:
                    #for idx in range(len(dictionary)):
                    #    token = dictionary[idx]
                    #    assert token in embed_dict
                    emb = Embedding(num_embeddings, pretrain_embedding_dim, padding_idx)
                else:
                    emb = Embedding(num_embeddings, embed_dim, padding_idx)
                utils.load_embedding(embed_dict, dictionary, emb)
            else:
                emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        embed_tokens = build_embedding(
                task.target_dictionary, cfg.encoder_embed_dim, getattr(cfg, "embed_path", None)
                #task.target_dictionary, cfg.embed_dim, None
            )
        mt_model_encoder = MTModelEncoder(cfg, task.target_dictionary, embed_tokens)
        if cfg.freeze_modules!="":
            freeze_names = cfg.freeze_modules.split(",")
            #for key, value in w2v_encoder.named_parameters():
            #   for freeze_name in freeze_names:
            #       if freeze_name in key:
            #           value.requires_grad = False
            for key, value in mt_model_encoder.named_parameters():
               for freeze_name in freeze_names:
                   if freeze_name in "encoder."+key:
                       value.requires_grad = False
               if "down_proj" in "encoder."+key:
                   value.requires_grad = True
               if "up_proj" in "encoder."+key:
                   value.requires_grad = True
        if cfg.tune_w2v_LNA:
            #utils.freeze_parameters(encoder, "wav2vec_model")
            for key, value in w2v_encoder.w2v_model.named_parameters():
               value.requires_grad = False
               for layer_index in range(16,24):
                   if "layer_norm" in key and str("."+str(layer_index)+".") in key :
                       value.requires_grad = True
                   elif "self_attn" in key and str("."+str(layer_index)+".") in key :
                       value.requires_grad = True
            logging.info("only tune the wav2vec layer normal and self-attn module")

        return cls(cfg, w2v_encoder, mt_model_encoder)

    def get_normalized_probs(self, net_output, log_probs, ctc_constrative=False):
        """Get normalized probabilities (or log probs) from a net's output."""
        if ctc_constrative:
            logits = net_output["wav2vec_out"]
        else:
            logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_logits(self, net_output):
        logits = net_output["encoder_out"]
        padding = net_output["padding_mask"]
        if padding is not None and padding.any():
            padding = padding.T
            logits[padding][...,0] = 0
            logits[padding][...,1:] = float('-inf')

        return logits

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x

class NormalAdapter(BaseFairseqModel):
    def  __init__(self, cfg: CmclAdapterTransformerConfig, padding_idx):
        super().__init__()
        self.subsample = Conv1dSubsampler(
            cfg.encoder_embed_dim + cfg.position_unit_size,
            cfg.conv_channels,
            cfg.encoder_embed_dim + cfg.position_unit_size,
            [int(k) for k in cfg.conv_kernel_sizes.split(",")],
        ) if cfg.use_cnn_module else None
        self.embed_scale = math.sqrt(cfg.encoder_embed_dim + cfg.position_unit_size)
        self.embed_positions = (
            PositionalEmbedding(
                getattr(cfg, "max_target_positions", 3000),
                cfg.encoder_embed_dim + cfg.position_unit_size,
                padding_idx,
            )
        ) if cfg.add_position_embed and not cfg.add_position_embed_after_ctc else None
        
        if cfg.relative_attn:
            self.relative_attn = cfg.relative_attn
            self.embed_positions = RelPositionalEncoding(
                getattr(cfg, "max_target_positions", 3000), cfg.encoder_embed_dim
            )
            from omegaconf import open_dict
            import copy
            cfg=copy.deepcopy(cfg)
            with open_dict(cfg):
                cfg.encoder_attention_type="rel_selfattn"
        else:
            self.relative_attn =False
        self.layers=nn.ModuleList(
            [ConformerEncoderLayer(cfg.encoder_embed_dim + cfg.position_unit_size,
                cfg.encoder_ffn_embed_dim,
                cfg.encoder_attention_heads,
                cfg.dropout,
                cfg.fp16,
                cfg.cnn_module_kernel,
            ) for _ in range(cfg.adapter_layers)]
        )

    def forward(self, x, padding_mask):
        if self.subsample:
            input_length = (1 - padding_mask.int()).sum(dim=1)
            x, output_lengths = self.subsample(x, input_length)
            padding_mask = lengths_to_padding_mask(output_lengths)
        else:
            x = x.transpose(0, 1)
        if self.embed_positions:
            if self.relative_attn:
                positions = self.embed_positions(x)
            else:
                positions = self.embed_positions(padding_mask).transpose(0, 1)
                x = self.embed_scale * x
                x += positions
        for layer in self.layers:
            if self.relative_attn:
                x, _ = layer(x, padding_mask, pos_emb=positions)
            else:
                x, _ = layer(x, padding_mask)
        return x, padding_mask

class SemanticAdapter(BaseFairseqModel):
    def  __init__(self, cfg: CmclAdapterTransformerConfig, padding_idx):
        super().__init__()
        #self.embed_scale = math.sqrt(cfg.encoder_embed_dim)
        self.embed_positions = (
            PositionalEmbedding(
                getattr(cfg, "max_target_positions", 3000),
                cfg.encoder_embed_dim,
                padding_idx,
            )
        ) if cfg.add_position_embed and cfg.add_position_embed_after_ctc else None
        self.encoder_normalize_before=cfg.encoder_normalize_before
        self.contrastive_mask_prob = getattr(cfg, "contrastive_mask_prob", 0.0)
        if self.contrastive_mask_prob > 0:
            self.mask_emb = nn.Parameter(
                torch.FloatTensor(cfg.encoder_embed_dim).zero_()
            )
        if cfg.transfer_proj:
            self.transfer_proj = nn.Linear(cfg.encoder_embed_dim, cfg.encoder_embed_dim, bias=False)
        else:
            self.transfer_proj = None
        self.layers=nn.ModuleList(
            [TransformerEncoderLayer(cfg) for _ in range(cfg.sead_layers)]
        )
        if self.encoder_normalize_before:
            self.layer_norm = LayerNorm(cfg.encoder_embed_dim)
 
        self.conv_list = None
        self.conv_norm_list = None
         

    def forward(self, x, padding_mask):
        if self.embed_positions:
            positions = self.embed_positions(padding_mask).transpose(0, 1)
            #x = self.embed_scale * x
            x = x + positions
        if self.contrastive_mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.contrastive_mask_prob,
                1,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb

        #with open("st_attn_weight","a") as f:
        #    f.write(str(list(x.shape)[0])+"\n")
        for index, layer in enumerate(self.layers):
            if self.conv_list and index < len(self.conv_list):
                residual = x
                x = self.conv_norm_list[index](x)
                x = self.conv_list[index](x.transpose(0, 1), padding_mask)
                x = x.transpose(0, 1)
                x = residual + x
            if self.transfer_proj is not None:
                x = layer(x, padding_mask, transfer_proj = self.transfer_proj)
            else:
                x,attn = layer(x, padding_mask)
            #with open("st_attn_weight","a") as f:
            #    atten_weight = [str(item) for item in attn.squeeze(0).flatten().tolist()]
            #    f.write(' '.join(atten_weight)+"\n")
        if self.encoder_normalize_before:
            x = self.layer_norm(x)
        return x, padding_mask
       

class Wav2vecEncoder(FairseqEncoder):
    def __init__(self, cfg: CmclAdapterTransformerConfig, tgt_dict=None):

        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }
        
        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = convert_namespace_to_omegaconf(w2v_args)
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )

        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model, from_checkpoint=True)
        model.remove_pretraining_modules()
        subsample_state = {}
        layers_state = {}

        super().__init__(task.source_dictionary)
        d = w2v_args.model.encoder_embed_dim
         
        if d!=cfg.encoder_embed_dim + cfg.position_unit_size:
            self.compress_ffn = nn.Linear(d, cfg.encoder_embed_dim + cfg.position_unit_size, True)
            nn.init.xavier_uniform_(self.compress_ffn.weight)
            nn.init.constant_(self.compress_ffn.bias, 0.0)
        else:
            self.compress_ffn = None

        if cfg.use_ctc_loss:
            if tgt_dict is not None:
                self.proj = Linear(cfg.encoder_embed_dim, len(tgt_dict), bias=False)
                if cfg.embed_path != "":

                    def load_embedding_weight(dictionary, embed_dim, path):
                        num_embeddings = len(dictionary)
                        padding_idx = dictionary.pad()
                        logging.info("load pretrain embeddings: {} to init proj".format(path))
                        embed_dict = utils.parse_embedding(path)
                        pretrain_embedding_dim=list(list(embed_dict.values())[0].size())[0]
                        assert pretrain_embedding_dim == embed_dim, "the pretrain embedding size should be same as hidden size"
                        emb = Embedding(num_embeddings, embed_dim, padding_idx)
                        utils.load_embedding(embed_dict, dictionary, emb)
                        return emb.weight

                    self.proj.weight = load_embedding_weight(tgt_dict, cfg.encoder_embed_dim, cfg.embed_path)

            elif getattr(cfg, "decoder_embed_dim", d) != d:
                self.proj = Linear(d, cfg.decoder_embed_dim, bias=False)
            else:
                self.proj = None
        else:
            self.proj = None

        #del model.w2v_encoder.proj
        if "wav2vec_small" not in cfg.w2v_path and "hubert" not in cfg.w2v_path:
            param_state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            if param_state is not None and not cfg.no_pretrained_weights:
                #model.load_state_dict(state["model"], strict=True)
                for key in list(param_state['model'].keys()):
                   w = param_state['model'].pop(key)
                   if key.startswith('w2v_encoder.w2v_model') and "w2v_encoder.proj" not in key:
                       new_key = key.replace('w2v_encoder.w2v_model.', '')
                       param_state['model'][new_key] = w
                   if key.startswith('w2v_encoder.subsample'):
                       new_key = key.replace('w2v_encoder.subsample.', '') 
                       subsample_state[new_key] = w
                   if key.startswith('w2v_encoder.layers'):
                       new_key = key.replace('w2v_encoder.layers.', '')
                       layers_state[new_key] = w
                   if key.startswith('w2v_encoder.proj') and self.proj is not None:
                       self.proj.weight.data = w
                       logging.info("load pretrain ctc proj state")
                model.load_state_dict(param_state["model"], strict=True)
        elif "hubert" in cfg.w2v_path:
            logging.info("load pretrained hubert")
            model.load_state_dict(state["model"], strict=True)
        else:
            logging.info("load pretrained small wav2vec")
            model.load_state_dict(state["model"], strict=True)
        
        self.w2v_model = model
        self.noad = NormalAdapter(cfg, tgt_dict.pad())
        if cfg.sead_layers: 
            self.sead = SemanticAdapter(cfg, tgt_dict.pad())
        else:
            self.sead = None
        if subsample_state != {}:
            self.noad.subsample.load_state_dict(subsample_state, strict=True)
            logging.info("load normal adapter subsample state")
        if layers_state != {}:
            self.noad.layers.load_state_dict(layers_state, strict=True)
            logging.info("load normal adapter layer state")

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0


    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, tbc=True, **kwargs):
        if source.shape != padding_mask.shape:
            padding_mask = lengths_to_padding_mask(padding_mask)
 
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            wav_feature, padding_mask = self.w2v_model.extract_features(**w2v_args)
        if self.compress_ffn:
            wav_feature = self.compress_ffn(wav_feature)
        #print(wav_feature.shape,end=" ")
        if self.noad:
            wav_feature, padding_mask = self.noad(wav_feature, padding_mask)
        elif tbc:
            # B x T x C -> T x B x C
            wav_feature = wav_feature.transpose(0, 1)
        x = self.final_dropout(wav_feature)
        wav_feature=x
        if self.proj:
            wav_out = self.proj(x)
        else:
            wav_out = x
        if self.sead:
            x, padding_mask = self.sead(x, padding_mask)
        else:
            x=wav_out
        return {
            "encoder_out": x,  # T x B x C
            "wav2vec_out": wav_out,
            "wav2vec_feature": wav_feature,
            "encoder_padding_mask": padding_mask,  # T x B
            "padding_mask": padding_mask,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict

class MTModelEncoder(TransformerEncoder):
    def __init__(self, cfg: CmclAdapterTransformerConfig, tgt_dict, embed_tokens):
        arg_overrides = {
            "dropout": cfg.dropout,
        }
        if cfg.mt_model_args is None and getattr(cfg, "mt_model_path", None):
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.mt_model_path, arg_overrides)
            mt_model_args = state.get("cfg", None) 
            assert  mt_model_args is not None,("Miss mt model args") 
            super().__init__(mt_model_args.model, tgt_dict, embed_tokens)
        else:
            state=None
            super().__init__(cfg, tgt_dict, embed_tokens)
                
        #arg=copy.deepcopy()
        #arg.decoder_layers=arg.mbart_layers
        #arg.input_embed_dim = embed_tokens.embedding_dim

        embed_dim = embed_tokens.embedding_dim
        if state:
            for key in list(state['model'].keys()):
                w = state['model'].pop(key)
                if key.startswith('encoder') and "embed_tokens" not in key:
                    new_key = key.replace('encoder.', '')
                    state['model'][new_key] = w
            self.load_state_dict(state["model"], strict=False)
            logging.info("load pretrained encoder: {}".format(cfg.mt_model_path))
        self.filter_size = cfg.mt_model_filter_size
        if self.filter_size != 0:
            self.down_proj = nn.Linear(embed_tokens.embedding_dim, self.filter_size)
            self.up_proj = nn.Linear(self.filter_size, embed_tokens.embedding_dim)
    
    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):

        encoder_out = self.forward_scriptable(src_tokens,
                                       src_lengths,
                                       return_all_hiddens,
                                       token_embeddings)    
        if self.filter_size != 0:
            x = encoder_out["encoder_out"][0]
            x = self.down_proj(x)
            x = F.relu(x)
            x = self.up_proj(x)
            encoder_out["encoder_out"]=[x]
        return encoder_out

    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
    
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)
        
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        x = x.transpose(0, 1)

        encoder_states = []
        fc_results = []

        if return_all_hiddens:
            encoder_states.append(x)


        for layer in self.layers:
            lr = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )

            if isinstance(lr, tuple) and len(lr) == 2:
                x, fc_result = lr
            else:
                x = lr
                fc_result = None

            if return_all_hiddens and not torch.jit.is_scripting():
                assert encoder_states is not None
                encoder_states.append(x)
                fc_results.append(fc_result)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        src_lengths = (
            src_tokens.ne(self.padding_idx)
            .sum(dim=1, dtype=torch.int32)
            .reshape(-1, 1)
            .contiguous()
        )

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "fc_results": fc_results,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }




class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        cfg: CmclAdapterTransformerConfig,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
    ):
        super().__init__(dictionary)

        self.dropout = cfg.decoder_dropout
        self.share_input_output_embed = cfg.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = cfg.decoder_embed_dim
        self.output_embed_dim = cfg.decoder_embed_dim

        self.layerdrop = cfg.decoder_layerdrop

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = cfg.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                cfg.max_target_positions,
                embed_dim,
                padding_idx,
                learned=cfg.decoder_learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )

        # TODO: update this when transformer gets converted to dataclass configs
        transformer_cfg = copy.deepcopy(cfg)
        with open_dict(transformer_cfg):
            transformer_cfg.dropout = transformer_cfg.decoder_dropout
            transformer_cfg.attention_dropout = (
                transformer_cfg.decoder_attention_dropout
            )
            transformer_cfg.activation_dropout = (
                transformer_cfg.decoder_activation_dropout
            )

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerDecoderLayer(transformer_cfg, no_encoder_attn)
                for _ in range(transformer_cfg.decoder_layers)
            ]
        )

        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(
                torch.Tensor(len(dictionary), self.output_embed_dim)
            )
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if transformer_cfg.decoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        prev_output_tokens = prev_output_tokens.long()
        x, extra = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )
        x = self.output_layer(x)
        return x, extra

    def extract_features(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, attn, _ = layer(
                    x,
                    encoder_out["encoder_out"] if encoder_out is not None else None,
                    encoder_out["padding_mask"]
                    if encoder_out is not None
                    else None,
                    incremental_state,
                    self_attn_mask=self.buffered_future_mask(x)
                    if incremental_state is None
                    else None,
                )
                inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, {"attn": attn, "inner_states": inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        # project back to size of vocabulary
        if self.share_input_output_embed:
            return F.linear(features, self.embed_tokens.weight)
        else:
            return F.linear(features, self.embed_out)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
