#!/usr/bin/env python3

from argparse import Namespace
import logging
import math
import time
import os
from typing import Dict, List, Optional, Tuple
import contextlib

import torch
import torch.nn as nn
from fairseq import checkpoint_utils, utils, tasks
import torch.nn.functional as F
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import Embedding, TransformerDecoder
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
    ConformerEncoderLayer,
    MultiheadAttention,
)
from collections import OrderedDict
from fairseq.models.speech_to_text.s2t_transformer import Conv1dSubsampler, TransformerDecoderScriptable, S2TTransformerEncoder
from fairseq.models.wav2vec import (
    Wav2Vec2Model, Wav2VecCtc,Wav2Vec2Config
)
from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
#from fairseq.modes.s2t_transformer_w2v2 import S2T_W2V2_ConformerEncoder 
from fairseq.models.wav2vec.cmcl_adapter_transformer import NormalAdapter, SemanticAdapter, MTModelEncoder, Wav2vecEncoder
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

#from .hdfs_utils import torchHLoad
from torch import Tensor

logger = logging.getLogger(__name__)


@register_model("s2t_joint")
class S2TJoint(BaseFairseqModel):
    """Adapted Transformer model (https://arxiv.org/abs/1706.03762) for
    speech-to-text tasks. The Transformer encoder/decoder remains the same.
    A trainable input subsampler is prepended to the Transformer encoder to
    project inputs into the encoder dimension as well as downsample input
    sequence for computational efficiency."""

    def __init__(self, acoustic_encoder, textual_encoder, decoder, task_net):
        super().__init__()
        self.acoustic_encoder = acoustic_encoder
        self.textual_encoder = textual_encoder
        self.decoder = decoder
        self.task_net = task_net

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # input
        parser.add_argument("--w2v2-model-path", type=str, metavar="N",
                            help="path/to/wav2vec/model, support hdfs")
        parser.add_argument(
            "--conv-kernel-sizes",
            type=str,
            metavar="N",
            help="kernel sizes of Conv1d subsampling layers",
        )
        parser.add_argument(
            "--conv-channels",
            type=int,
            metavar="N",
            help="# of channels in Conv1d subsampling layers",
        )
        ## Transformer
        parser.add_argument(
            "--activation-fn",
            type=str,
            default="relu",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--share-two-encoders",
            action="store_true",
            help="share the parameter of acoustic and textual encoders",
        )
        parser.add_argument(
            "--transfer-proj",
            type=bool,
            default=False,
            help="use proj to relax the parameter of acoustic and textual encoders",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--load-encoder-layers-from",
            type=str,
            metavar="STR",
            help="model to take encoder layers component weights from (for initialization)",
        )
        parser.add_argument(
            "--load-pretrained-decoder-from",
            type=str,
            metavar="STR",
            help="model to take decoder weights from (for initialization)",
        )
        parser.add_argument(
            "--macaron-style",
            default=False,
            type=bool,
            help="whether to use macaron style for positionwise layer",
        )
        ## Attention
        parser.add_argument(
            "--zero-triu",
            default=False,
            type=bool,
            help="If true, zero the uppper triangular part of attention matrix.",
        )
        # Relative positional encoding
        parser.add_argument(
            "--rel-pos-type",
            type=str,
            default="legacy",
            choices=["legacy", "latest"],
            help="Whether to use the latest relative positional encoding or the legacy one."
                 "The legacy relative positional encoding will be deprecated in the future."
                 "More Details can be found in https://github.com/espnet/espnet/pull/2816.",
        )
        # CNN module
        parser.add_argument(
            "--use-cnn-module",
            default=False,
            type=bool,
            help="Use convolution module or not",
        )
        parser.add_argument(
            "--cnn-module-kernel",
            default=31,
            type=int,
            help="Kernel size of convolution module.",
        )
        parser.add_argument(
            "--text-conv-kernel",
            default=0,
            type=int,
            help="Kernel size of convolution module used in textual encoder",
        )
        parser.add_argument(
            "--encoder-freeze-module",
            type=str,
            metavar="STR",
            help="freeze the module of the encoder",
        )
        parser.add_argument(
            "--decoder-freeze-module",
            type=str,
            metavar="STR",
            help="freeze the module of the decoder",
        )
        parser.add_argument(
            '--decoder-embed-path', type=str, metavar='STR',
            help='path to pre-trained decoder embedding'
        )
        parser.add_argument(
            '--cluster-embed-path', type=str, 
            metavar='STR', default="",
            help='path to cluster embedding'
        )
        parser.add_argument(
            "--use-w2v-ctc",
            action="store_true",
            help="use ctc loss to update wav2vec",
        )
        parser.add_argument(
            "--share-ctc-embed",
            action="store_true",
            help="share the ctc proj and encoder embed",
        )
        parser.add_argument(
            "--tune-w2v-LNA",
            action="store_true",
            help="update wav2vec layer norm",
        )
        parser.add_argument(
            "--tune-mbart-LNA",
            action="store_true",
            help="update mbart layer norm",
        )
        parser.add_argument(
            "--tune-encoder-LNA",
            action="store_true",
            help="update encoder layer norm",
        )
        parser.add_argument(
            "--freeze-textual-encoder",
            action="store_true",
            help="freeze textual encoder",
        )
        parser.add_argument(
            "--drop-w2v-last-layer",
            action="store_true",
            help="dropout last w2v model layer to avoid over-fitting",
        )
        parser.add_argument(
            "--layerdrop",
            type=float,
            default=0.0,
            help= "probability of dropping a layer in wav2vec 2.0"
        )
        parser.add_argument(
            "--feature-grad-mult",
            type=float,
            default=0.0, 
            help="reset feature grad mult in wav2vec 2.0 to this"
        )

        parser.add_argument(
            "--adapter-dim",
            type=int,
            metavar="N",
            help="adapter dimension",
        )
        parser.add_argument(
            "--adapter-dropout",
            type=float,
            metavar="D",
            help="dropout probability for the adapter",
        )
        parser.add_argument(
            "--dropout-input",
            type=float,
            default=0.0
        )
        parser.add_argument(
            "--final-dropout", 
            type=float,
            default=0.0,
            help= "dropout after transformer and before final projection"
        )
        parser.add_argument(
            "--mask-length",
            type=int, 
            default=10, 
            help="repeat the mask indices multiple times"
        )
        parser.add_argument(
            "--mask-selection",
            default="static", 
            help="how to choose masks"
        )
        parser.add_argument(
            "--mask-other",
            type=float,
            default=0,
            help="secondary mask argument (used for more complex distributions), see help in compute_mask_indices"
        )
        parser.add_argument(
            "--no-mask-overlap",
            type=bool,
            default=False,
            help="whether to allow masks to overlap"
        )

        ## channel masking
        parser.add_argument(
            "--mask-channel-length",
            type=int,
            default=10, 
            help="length of the mask for features (channels)"
        )
        parser.add_argument(
            "--mask-channel-selection",
            default="static",
            help= "how to choose mask length for channel masking"
        )
        parser.add_argument(
            "--mask-channel-other",
            type=float,
            default=0,
            help="secondary mask argument (used for more complex distributions), see help in compute_mask_indicesh"
        ) 
        parser.add_argument(
            "--no-mask-channel-overlap",
            type=bool,
            default=False, 
            help= "whether to allow channel masks to overlap"
        )
        parser.add_argument(
            "--adapter-layers",
            type=int,
            default=0,
            help="conformer layers of normal adapter"
        )
        parser.add_argument(
            "--sead-layers",
            type=int,
            default=0,
            help="conformer layers of semantic adapter"
        )
        parser.add_argument(
            "--additional-adapter",
            action="store_true",
            help='add an additional adapter between encoder and decoder'
        )
        parser.add_argument(
            "--apply-mask",
            action="store_true",
            help="apply mask on audio feature",
        )
        parser.add_argument(
            "--mask-prob",
            type=float,
            default=0.0,
            metavar="D",
            help="probability of replacing a token with mask (normalized by length)",
        )
        parser.add_argument(
            "--mask-channel-prob",
            type=float,
            default=0.0,
            metavar="D",
            help="probability of replacing a feature with 0",
        )
        parser.add_argument(
            "--normalize",
            type=bool,
            default=False,
            help="normalize the speech input",
        )
        parser.add_argument(
            "--use-ctc-loss",
            type=bool,
            default=False,
            help="using ctc loss"
        )
        parser.add_argument(
            "--max-position-ctc",
            type=int,
            default=0,
            help="using ctc to predict the position rather than word, set 0 to ban this"
        )
        parser.add_argument(
            "--use-ctc-shrink",
            type=bool,
            default=False,
            help="using ctc prediction to shrink sequence"
        )
        parser.add_argument(
            "--avg-shrink",
            type=bool,
            default=False,
            help="using average feature when shrinking sequence"
        )
        parser.add_argument(
            "--lookback",
            type=bool,
            default=False,
            help="using average feature when shrinking sequence"
        )
        parser.add_argument(
            "--position-unit-size",
            type=int,
            default=0,
            help="the unit size to predict the speech position"
        )
        parser.add_argument(
            "--latent-temp",
            type=tuple,
            default=(1, 0.1, 0.999995),
            help="temperature for latent variable sampling."
        )
        parser.add_argument(
            "--embed-path",
            type=str,
            default="",
            help="path to word embed"
        )
        parser.add_argument(
            "--w2v-path",
            type=str,
            default=None,
            help="path to wav2vec 2.0 model"
        )
        parser.add_argument(
            "--add-position-embed",
            type=bool,
            default=True,
            help="add position embed"
        )
        parser.add_argument(
            "--add-position-embed-after-ctc",
            type=bool,
            default=False,
            help="add position embed after ctc loss"
        )
        parser.add_argument(
            "--relative-attn",
            type=bool,
            default=False,
            help="use relative position attention"
        )
        parser.add_argument(
            "--freeze-finetune-updates",
            type=int,
            default=0, 
            help="dont finetune wav2vec for this many updates"
        )
        parser.add_argument(
            "--mt-model-path", 
            type=str,
            help="path to pretrained mt model"
        )
        parser.add_argument(
            "--mt-model-filter-size",
            type=int,
            default=0, 
            help="ffn filter size of model"
        ) 
        parser.add_argument(
            "--w2v-args",
            default=None
        )
        parser.add_argument(
            "--mt-model-args",
            default=None
        )

    @classmethod
    def build_acoustic_encoder(cls, args, task=None, embed_tokens=None):
        encoder = AcousticEncoder(args, task.target_dictionary, embed_tokens)
        return encoder

    @classmethod
    def build_textual_encoder(cls, args, task=None, embed_tokens=None):
        encoder = MTModelEncoder(args, task.target_dictionary, embed_tokens)
        if getattr(args, "load_pretrained_mt_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from, strict=True
            )
            logger.info(
                f"load pretrained mt encoder from: "
                f"{args.load_pretrained_mt_encoder_from}"
            )
        return encoder


    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()

    @classmethod
    def build_decoder(cls, cfg, task, embed_tokens):
        if getattr(cfg, "mt_model_path", None):
            arg_overrides = {
                "dropout": cfg.dropout,
            }
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.mt_model_path, arg_overrides)
            mt_model_args = state.get("cfg", None)
            assert  mt_model_args is not None,("Miss mt model args")
            decoder = TransformerDecoderScriptable(mt_model_args.model, task.target_dictionary, embed_tokens)
         
            for key in list(state['model'].keys()):
                w = state['model'].pop(key)
                if key.startswith('decoder') :
                    new_key = key.replace('decoder.', '')
                    state['model'][new_key] = w
            decoder.load_state_dict(state["model"], strict=True)
            logging.info("load pretrained decoder: {}".format(cfg.mt_model_path))
        else:
            decoder = TransformerDecoderScriptable(cfg, task.target_dictionary, embed_tokens)
            if getattr(cfg, "load_pretrained_decoder_from", None):
                logger.info(
                    f"loaded pretrained decoder from: "
                    f"{cfg.load_pretrained_decoder_from}"
                )
                decoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=decoder, checkpoint=cfg.load_pretrained_decoder_from, strict=True
                )

        return decoder


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            if path:
                logging.info("load pretrained embeddings: {}".format(path))
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

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim, getattr(args, "decoder_embed_path", None)
            #task.target_dictionary, args.decoder_embed_dim
        )

        #encoder = cls.build_encoder(args, task, decoder_embed_tokens)
        acoustic_encoder = cls.build_acoustic_encoder(args, task, decoder_embed_tokens)
        textual_encoder = cls.build_textual_encoder(args, task, decoder_embed_tokens)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        task_net = TaskNetwork(512,2)
        
        if getattr(args, "encoder_freeze_module", None):
            utils.freeze_parameters(encoder, args.encoder_freeze_module)
            logging.info("freeze the encoder module: {}".format(args.encoder_freeze_module))

        if getattr(args, "decoder_freeze_module", None):
            utils.freeze_parameters(decoder, args.decoder_freeze_module)
            logging.info("freeze the decoder module: {}".format(args.decoder_freeze_module))

        if hasattr(args, 'tune_w2v_LNA'):
            #utils.freeze_parameters(encoder, "wav2vec_model")
            for key, value in acoustic_encoder.w2v_model.named_parameters():
               if "layer_norm" in key:
                   value.requires_grad = True
               elif "self_attn" in key:
                   value.requires_grad = True
               else:
                   value.requires_grad = False
            logging.info("only tune the wav2vec layer normal and self-attn module")
        if hasattr(args, 'tune_mbart_LNA'):
            for key, value in decoder.named_parameters():
               if "layer_norm" in key:
                   value.requires_grad = True
               elif "encoder_attn" in key:
                   value.requires_grad = True
               else:
                   value.requires_grad = False
            logging.info("only tune the decoder layer normal and encoder-attn module")
        if hasattr(args, 'tune_encoder_LNA'):
            utils.freeze_parameters(textual_encoder.layers, "layers")
            for key, value in textual_encoder.layers.named_parameters():
               if "layer_norm" in key:
                   value.requires_grad = True
               elif "self_attn" in key:
                   value.requires_grad = True
               else:
                   value.requires_grad = False
            logging.info("only tune the encoder layer normal and self-attn module")
        if hasattr(args, 'freeze_textual_encoder'):
            for key, value in textual_encoder.layers.named_parameters():
                   value.requires_grad = False
            logging.info("freeze the texutal encoder layers")
        if hasattr(args, 'share_two_encoders'):
            logging.info("share the sematic adapter and textual encoder")
            if args.sead_layers==textual_encoder.num_layers:
                acoustic_encoder.sead.layers=textual_encoder.layers
            else:
                for layer_i in range(textual_encoder.num_layers):
                    textual_encoder.layers[layer_i] = acoustic_encoder.sead.layers[args.sead_layers - textual_encoder.num_layers + layer_i]
            if textual_encoder.conv_list is not None:
                acoustic_encoder.sead.conv_list = textual_encoder.conv_list
                acoustic_encoder.sead.conv_norm_list = textual_encoder.conv_norm_list
        else:
            #for i in range(1,len(acoustic_encoder.sead.layers)):
            logging.info("share some parts between the sematic adapter and textual encoder")
            assert args.sead_layers==textual_encoder.num_layers, "the number of encoder layers should be the same"
            for i in range(0,len(acoustic_encoder.sead.layers)):
                acoustic_encoder.sead.layers[i].self_attn = textual_encoder.layers[i].self_attn
                acoustic_encoder.sead.layers[i].self_attn_layer_norm = textual_encoder.layers[i].self_attn_layer_norm
                acoustic_encoder.sead.layers[i].fc1 = textual_encoder.layers[i].fc1
                acoustic_encoder.sead.layers[i].fc2 = textual_encoder.layers[i].fc2
        return cls(acoustic_encoder, textual_encoder, decoder, task_net)

    def get_acoustic_projection_L2_norm(self):
        """Get L2 norm of projection to avoid overfitting"""
        return self.acoustic_encoder.pos_proj.weight.norm()

    # ************/ Add L2 norm /************ #
    def get_acoustic_embedding_L2_norm(self, weight=1.0):
        return self.acoustic_encoder.proj.weight.norm() * weight

    @classmethod
    def get_acoustic_normalized_probs(self, net_output, log_probs, ctc_contrastive=False, cluster=False):
        """Get normalized probabilities (or log probs) from a net's output."""
        if ctc_contrastive:
            logits = net_output["wav2vec_out"][0]
        elif cluster:
            logits = net_output["cluster_wav2vec_out"][0]
        else:
            logits = net_output["encoder_out"][0]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)
    
    def get_acoustic_normalized_pos_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output["pos_wav2vec_out"][0]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1) 

    def forward(self, src_tokens, src_lengths,
                prev_output_tokens, **extra_args):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        return decoder_out

class LookBackModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder_attn = MultiheadAttention(
            cfg.encoder_embed_dim,
            cfg.encoder_attention_heads,
            kdim=cfg.encoder_embed_dim,
            vdim=cfg.encoder_embed_dim,
            dropout=cfg.dropout,
            encoder_decoder_attention=True
        )
        self.atten_layer_norm = LayerNorm(cfg.encoder_embed_dim)
        self.fc1 = nn.Linear(cfg.encoder_embed_dim, cfg.encoder_ffn_embed_dim)
        self.fc2 = nn.Linear(cfg.encoder_ffn_embed_dim, cfg.encoder_embed_dim)
        self.activation_fn = nn.SiLU() #utils.get_activation_fn(activation="swish")
        self.ffn_layer_norm = LayerNorm(cfg.encoder_embed_dim)
        self.lb_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, wav_feature, bf_shrink_padding_mask):

        residual = x
        x, _ = self.encoder_attn(
            query=x,
            key=wav_feature,
            value=wav_feature,
            key_padding_mask=bf_shrink_padding_mask,
            incremental_state=None,
            static_kv=True,
            need_weights=False,
            #attn_mask=padding_mask,
        )
        x += residual
        x = self.lb_dropout(x)
        x = self.atten_layer_norm(x)
        residual = x
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        x += residual
        x = self.lb_dropout(x)
        x = self.ffn_layer_norm(x)
        return x

class AcousticEncoder(Wav2vecEncoder):
    def __init__(self, cfg, tgt_dict=None, embed_tokens=None):
        super().__init__(cfg, tgt_dict)  
        max_position_ctc = int(getattr(cfg, "max_position_ctc", 0))
        self.use_ctc_shrink = cfg.use_ctc_shrink
        self.avg_shrink = getattr(cfg, "avg_shrink", False)
        if self.use_ctc_shrink:
            self.shrink_layer_norm = LayerNorm(cfg.encoder_embed_dim)
            if getattr(cfg, "lookback", False):
                self.lbm = LookBackModule(cfg)
            else:
                self.lbm = None
        self.max_temp, self.min_temp, self.temp_decay = cfg.latent_temp
        self.position_unit_size = getattr(cfg, "position_unit_size", 0)
        if max_position_ctc == 0:
            cluster_embed_path = getattr(cfg, "cluster_embed_path", "")
            if cluster_embed_path != "":
                logging.info("load cluster embedding as ctc proj: {}".format(cfg.cluster_embed_path))
                embed_dict = utils.parse_embedding(cluster_embed_path)
                embed_weight = torch.stack(list(embed_dict.values()))
                self.cluster_proj = nn.Linear(embed_weight.shape[1], embed_weight.shape[0], bias=False)
                self.cluster_proj.weight = nn.Parameter(embed_weight)
                self.adapter = TransformerEncoderLayer(cfg)
                #self.adapter=StableAdapter(cfg.encoder_embed_dim)
            else:
                self.cluster_proj=None
            if getattr(cfg, "share_ctc_embed", False):
                self.proj.weight = embed_tokens.weight
            else:
                self.proj.weight.data = embed_tokens.weight.data
            if getattr(cfg, "decoder_embed_path", None):
                logging.info("load pretrained embedding as ctc proj: {}".format(cfg.decoder_embed_path))
        else:
            if self.position_unit_size != 0:
                self.pos_proj = nn.Linear(self.position_unit_size, max_position_ctc, bias=False)
            else:
                self.pos_proj = nn.Linear(cfg.encoder_embed_dim, max_position_ctc, bias=False)
 
    def forward(self, src_tokens, src_lengths, tbc=True, mixup_rate=0.0, mixup_for_whole_model=False, textual_encoder=None, update_num=None, **kwargs):
        padding_mask = lengths_to_padding_mask(src_lengths)

        w2v_args = {
            "source": src_tokens,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            #wav_feature, padding_mask = self.w2v_model.extract_features(**w2v_args) 
            res = self.w2v_model.extract_features(**w2v_args)
            if isinstance(res, tuple):
                wav_feature, padding_mask = res
            elif isinstance(res, dict):
                wav_feature = res["x"]
                padding_mask = res["padding_mask"]
            else:
                raise TypeError("Retuen type is not correct")
               
 
        if self.compress_ffn:
            wav_feature = self.compress_ffn(wav_feature)
        if self.noad:
            wav_feature, padding_mask = self.noad(wav_feature, padding_mask)
        elif tbc:
            # B x T x C -> T x B x C
            wav_feature = wav_feature.transpose(0, 1)
        bf_shrink_padding_mask = padding_mask
        if self.position_unit_size > 0:
            wav_feature, pos_feature = wav_feature.split([wav_feature.shape[-1] - self.position_unit_size, self.position_unit_size], -1)
        x = self.final_dropout(wav_feature)
        if self.cluster_proj:
            cluster_wav_out = self.cluster_proj(x)
            x=self.adapter(x, padding_mask)
        else:
            cluster_wav_out = None
        if self.proj:
            if self.position_unit_size > 0:
                pos_wav_out = self.pos_proj(pos_feature)
            else:
                pos_wav_out = None
            wav_out = self.proj(x)
        else:
            wav_out = x

        if not self.training and "task" in kwargs.keys() and kwargs["task"] == "asr":
            return {
                "encoder_out": [wav_out],  # T x B x C
                "encoder_padding_mask": [padding_mask],  # T x B
            }
        
        # add mixup
        x_mixup = None
        gen_mixup_rate = None
        if self.use_ctc_shrink and ft:
            wav_feature = wav_feature.transpose(0, 1)
            lprobs = self.compute_inner_ctc_prob(wav_out.transpose(0, 1))
            #shrink_mask = (lprobs.argmax(dim=-1) != 0)
            
            tokens = lprobs.argmax(dim=-1)
            #print(tokens.tolist())
            shrink_mask = tokens.roll(1) != tokens
            #print(shrink_mask.shape)
            #print(shrink_mask[:,0]) 
            shrink_mask[:,0] = True
            #print(tokens[:,0],tokens.roll(1)[:,0]) 
            if pos_wav_out is not None:
                pos_lprobs = self.compute_inner_ctc_prob(pos_wav_out.transpose(0, 1))
                pos_shrink_mask = (pos_lprobs.argmax(dim=-1) != 0)
                shrink_mask = pos_shrink_mask | shrink_mask
            if cluster_wav_out is not None:
                cluster_lprobs = self.compute_inner_ctc_prob(cluster_wav_out.transpose(0, 1))
                #cluster_shrink_mask = (cluster_lprobs.argmax(dim=-1) != 0)
                
                cluster_tokens = cluster_lprobs.argmax(dim=-1)
                cluster_shrink_mask = tokens.roll(1) != tokens
               
                shrink_mask = cluster_shrink_mask | shrink_mask
            #nonzero_shrink_mask = (lprobs.argmax(dim=-1) != 0)
            #adding the noise
            #noisy_mask = torch.rand(shrink_mask.shape).cuda() > 0.9
            #noisy_mask[:,0] = False
            #shrink_mask = shrink_mask ^ noisy_mask
            #shrink_mask = nonzero_shrink_mask | shrink_mask
            shrink_mask = shrink_mask & (~padding_mask)
            lengths = shrink_mask.long().sum(-1)
            
            # add mixup
            tokens_after_shrink = None
            if self.avg_shrink:
                old_lengths = (~padding_mask).long().sum(-1)
                for i in range(wav_feature.size(0)):
                    old_lengths[i] += i *wav_feature.size(1)
                #print(old_lengths)
                flatten_feature = wav_feature.flatten(0,1)
                flatten_mask = shrink_mask.flatten()
                unique_index = flatten_mask.nonzero().squeeze()
                max_len = lengths.max()
                x = flatten_feature.new_zeros(wav_feature.size(0), max_len, wav_feature.size(-1)) 
                batch_u = 0
                for batch_i, length in enumerate(lengths):
                    l_index = batch_i * wav_feature.size(1)
                    for u_i in range(1, length):
                        x[batch_i][u_i - 1] = flatten_feature[l_index:unique_index[u_i+batch_u],:].mean(0)
                        l_index = unique_index[u_i]
                    #remove padding
                    x[batch_i][length - 1] = flatten_feature[l_index:old_lengths[batch_i],:].mean(0)
                    batch_u += length
                #print(old_lengths,lengths)
            elif lengths.min() != 0:
                max_len = lengths.max()
                shrink_2d = wav_feature[shrink_mask]
                x = shrink_2d.new_zeros(wav_feature.size(0), max_len, wav_feature.size(-1))
                #x = shrink_2d.new_ones(wav_feature.size(0), max_len, wav_feature.size(-1))
                # add mixup
                if mixup_rate != 0.0:
                    tokens_after_shrink = tokens.new_zeros(shrink_mask.shape[0], max_len)

                l_index = 0
                for i, v in enumerate(lengths):
                    x[i, :v] = shrink_2d[l_index:l_index+v]
                    l_index += v
                    # add mixup
                    if mixup_rate != 0.0:
                        tokens_after_shrink[i, :v] = tokens[i][shrink_mask[i]]

            padding_mask = lengths_to_padding_mask(lengths)

            # Disable the Mask
            real_padding_mask = padding_mask
            padding_mask = torch.zeros(padding_mask.shape).bool().to(padding_mask.device)                 # //need_check//

            wav_feature = wav_feature.transpose(0, 1)

            x = self.shrink_layer_norm(x.transpose(0, 1))
            if self.lbm:
                #print(x.shape, wav_feature.shape, bf_shrink_padding_mask.shape)
                x = self.lbm(x, wav_feature, bf_shrink_padding_mask)
                #print(x.mean(-1).tolist())
            #with open("st_token","a")as fo:
            #    l = [str(i) for i in tokens.tolist()]
            #    fo.write(" ".join(l)+"\n")
            #    l = [str(i) for i in x.mean(-1).tolist()]
            #    fo.write(" ".join(l)+"\n")

            # Mix-up
            assert not self.avg_shrink, "Avg shrink do not support mix-up"
            mixup_sent_rate = 0.3
            if mixup_rate != 0.0 and tokens_after_shrink is not None :                                          # //need_check//
                gen_mixup_rate = torch.zeros(tokens_after_shrink.shape[0], 1)
                if torch.rand(1) < mixup_sent_rate:         # do for whole batch
                    x_mixup = x if mixup_for_whole_model else x.clone()
                    if mixup_rate < 0:
                        gen_mixup_rate = 0.4 * torch.rand(tokens_after_shrink.shape[0]) + 0.1    # [bs]     0.1 ~ 0.5
                        gen_mixup_rate = gen_mixup_rate.unsqueeze(-1)
                    else:
                        gen_mixup_rate = torch.full(gen_mixup_rate.shape, mixup_rate)          # note kkq change here 08.09.2023
                        # gen_mixup_rate = gen_mixup_rate.unsqueeze(-1)
                    mixup_mask = torch.rand(tokens_after_shrink.shape) < gen_mixup_rate.repeat(1, tokens_after_shrink.shape[1])      # [bs, len]
                    _, text_embedding = textual_encoder.forward_embedding(tokens_after_shrink)
                    x_mixup = x_mixup.transpose(0, 1)           # [bs, len]
                    
                    x_mixup[mixup_mask] = text_embedding[mixup_mask]
                    
                    x_mixup = x_mixup.transpose(0, 1)           # [len, bs]
            else:
                # print("No Mixup")
                assert x_mixup is None, "BUG HERE! x_mixup should be None"
                pass

        #with open("len_baseline","a")as fo:
        #    fo.write(str(list(x.shape)[0])+"\n")
        wav_feature=x

        if self.sead:
            x, padding_mask = self.sead(x, padding_mask)
            if x_mixup is not None and not mixup_for_whole_model:
                x_mixup, padding_mask = self.sead(x_mixup, padding_mask)        # mix-up for at 
        else:
            x = wav_feature
        #with open("st_top","a")as fo:
        #    l = [str(i) for i in tokens.unique_consecutive().tolist()]
        #    fo.write(" ".join(l)+"\n")
        #    l = [str(i) for i in x.mean(-1).tolist()]
        #    fo.write(" ".join(l)+"\n")
        return {
            "encoder_out": [x],  # T x B x C
            "wav2vec_out": [wav_out],
            "pos_wav2vec_out": [pos_wav_out],
            "cluster_wav2vec_out": [cluster_wav_out],
            "wav2vec_feature": [wav_feature],
            "encoder_padding_mask": [padding_mask],  # T x B
            "padding_mask": [bf_shrink_padding_mask],
            "encoder_output_mixup": [x_mixup],   # mix-up for at 
            "mixup_rate": [gen_mixup_rate]
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [encoder_out["encoder_padding_mask"][0].index_select(0, new_order)]
        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            }

    def compute_inner_ctc_prob(self, ctc_logits, temperature=1.0):
        ctc_logits = ctc_logits.float() 
        if ctc_logits.shape[-1] == 1:
            pos_probs = torch.sigmoid(ctc_logits) 
            return torch.cat((1 - pos_probs, pos_probs),-1)
        if self.training: 
            curr_temp = max(self.max_temp * self.temp_decay**self.num_updates, self.min_temp)
            return F.gumbel_softmax(ctc_logits, tau=curr_temp, hard=True).type_as(ctc_logits)
            #return F.gumbel_softmax(ctc_logits, tau=curr_temp, hard=False).type_as(ctc_logits)
            #return utils.log_softmax(ctc_logits, dim=-1)
        else:
            scale_ctc_logits = ctc_logits / temperature
            return utils.log_softmax(scale_ctc_logits, dim=-1)
            #y_soft = utils.softmax(scale_ctc_logits, dim=-1)
            #index = y_soft.max(-1, keepdim=True)[1]
            #y_hard = torch.zeros_like(scale_ctc_logits).scatter_(-1, index, 1.0)
            #ret = y_hard - y_soft.detach() + y_soft
            #return ret

class TaskNetwork(nn.Module):
    def __init__(self, feature_dim, task_num, dropout=0.1):
        assert task_num != 1, "The number of task should not be only one"
        assert task_num == 2, "Only support the two task now"
        super().__init__()
        self.layer_norm = LayerNorm(feature_dim)
        self.down_proj = nn.Linear(feature_dim, int(0.5*feature_dim))
        self.up_proj = nn.Linear(int(0.5*feature_dim), feature_dim)
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.task_proj = nn.Linear(feature_dim, 1, bias=False)

    def forward(self, x, padding_mask=None):                                  # //need_check//
        # print(f"Task Net Input: {x.shape}")
        residual = x
        x = self.down_proj(x)
        x = F.relu(x)
        x = self.up_proj(x)
        x = self.dropout_module(x)
        x += residual
        x = self.layer_norm(x)
        x = x.transpose(0,1)    # [bs, len, dim=512]
        if padding_mask is None:
            x = x.mean(dim=1)
        else:
            # print("> 1. ", x.shape)
            lengths = (~padding_mask).long().sum(dim=1).unsqueeze(-1)   # [bs, 1]
            lengths[lengths==0.0] = 1   # 特殊值处理
            mask_3d = padding_mask.unsqueeze(-1).expand(x.shape)        # [bs, len, dim=512]
            x[mask_3d] = 0.0
            x = x.sum(dim=1) / lengths  # [bs, dim=512]
            # print("> 2. ", x.shape)
        x = x.squeeze()
        x = self.task_proj(x)
        task_probs = torch.sigmoid(x)
        return task_probs


@register_model_architecture(model_name="s2t_joint", arch_name="s2t_joint")
def base_architecture(args):

    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # Conformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)


@register_model_architecture("s2t_joint", "s2t_joint_s")
def s2t_w2v2_mabrt_sead_s(args):
    args.use_asr_finetune_w2v = getattr(args, "use_asr_finetune_w2v", False)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)


