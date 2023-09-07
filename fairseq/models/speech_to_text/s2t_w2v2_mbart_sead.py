#!/usr/bin/env python3

from argparse import Namespace
import logging
import math
import os
from typing import Dict, List, Optional, Tuple
import contextlib

import torch
import torch.nn as nn
from fairseq import checkpoint_utils, utils, tasks
import torch.nn.functional as F
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
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
)
from collections import OrderedDict
from fairseq.models.speech_to_text.s2t_transformer import Conv1dSubsampler, TransformerDecoderScriptable, S2TTransformerEncoder
from fairseq.models.wav2vec import (
    Wav2Vec2Model, Wav2VecCtc,Wav2Vec2Config
)
from fairseq.models.hubert import HubertModel
from fairseq.models.wav2vec.cmcl_adapter import NormalAdapter,SemanticAdapter
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
#from .hdfs_utils import torchHLoad
from torch import Tensor

logger = logging.getLogger(__name__)


@register_model("s2t_w2v2_mabrt_sead")
class S2TW2V2MBARTSEAD(FairseqEncoderDecoderModel):
    """Adapted Transformer model (https://arxiv.org/abs/1706.03762) for
    speech-to-text tasks. The Transformer encoder/decoder remains the same.
    A trainable input subsampler is prepended to the Transformer encoder to
    project inputs into the encoder dimension as well as downsample input
    sequence for computational efficiency."""

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

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
        # Transformer
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
        # Attention
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
            help='path to pre-trained decoder embedding')
        parser.add_argument(
            "--use-w2v-ctc",
            action="store_true",
            help="use ctc loss to update wav2vec",
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
            "--drop-w2v-last-layer",
            action="store_true",
            help="dropout last w2v model layer to avoid over-fitting",
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
            "--adapter-model-path",
            type=str,
            metavar='STR',
            help='path to pre-trained semantic adapter'
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

    @classmethod
    def build_encoder(cls, args, task=None, embed_tokens=None):
        encoder = S2T_W2V2_ConformerEncoder(args, task, embed_tokens)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        decoder = TransformerDecoderScriptable(args, task.target_dictionary, embed_tokens)
        if getattr(args, "load_pretrained_decoder_from", None):
            logger.info(
                f"loaded pretrained decoder from: "
                f"{args.load_pretrained_decoder_from}"
            )
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from, strict=False, embed_tokens=embed_tokens.weight
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

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim, getattr(args, "decoder_embed_path", None)
            #task.target_dictionary, args.decoder_embed_dim
        )

        encoder = cls.build_encoder(args, task, decoder_embed_tokens)
        decoder = cls.build_decoder(
            args, task, decoder_embed_tokens)
        
        if getattr(args, "encoder_freeze_module", None):
            utils.freeze_parameters(encoder, args.encoder_freeze_module)
            logging.info("freeze the encoder module: {}".format(args.encoder_freeze_module))

        if getattr(args, "decoder_freeze_module", None):
            utils.freeze_parameters(decoder, args.decoder_freeze_module)
            logging.info("freeze the decoder module: {}".format(args.decoder_freeze_module))

        if hasattr(args, 'tune_w2v_LNA'):
            #utils.freeze_parameters(encoder, "wav2vec_model")
            for key, value in encoder.wav2vec_model.named_parameters():
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
               elif "embed_tokens" in key or "output_projection" in key:
                   pass
               else:
                   value.requires_grad = False
            logging.info("only tune the decoder layer normal and encoder-attn module")
        if hasattr(args, 'tune_encoder_LNA'):
            utils.freeze_parameters(encoder.layers, "layers")
            for key, value in encoder.layers.named_parameters():
               if "layer_norm" in key:
                   value.requires_grad = True
               elif "self_attn" in key:
                   value.requires_grad = True
               else:
                   value.requires_grad = False
            logging.info("only tune the encoder layer normal and self-attn module")
        return cls(encoder, decoder)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

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

class StableAdapter(nn.Module):
    """
    Adapter for model finetuning, as described in:
    https://arxiv.org/pdf/1909.08478.pdf
    """
    def __init__(self, feature_dim,  dropout=0.1):
        super().__init__()
        self.layer_norm = LayerNorm(feature_dim)
        self.down_proj = nn.Linear(feature_dim, int(0.5*feature_dim))
        self.up_proj = nn.Linear(int(0.5*feature_dim), feature_dim)
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = F.relu(x)
        x = self.up_proj(x)
        x = self.dropout_module(x)
        x += residual
        x = self.layer_norm(x)
        return x

class S2T_W2V2_ConformerEncoder(S2TTransformerEncoder):
    """Speech-to-text Transformer encoder that consists of input wav2vec2Encoder, subsampler and
    Transformer encoder."""

    def __init__(self, args, task=None, embed_tokens=None):
        super().__init__(args, task, embed_tokens)

        #print( state["cfg"].model.w2v_args.model)
        assert args.adapter_model_path is not None
        self.adapter_model_path = args.adapter_model_path
        self.max_source_positions = args.max_source_positions
        self.apply_mask = getattr(args, 'apply_mask', False)
        self.additional_adapter = getattr(args, 'additional_adapter', False)
        self.adapter_dropout=getattr(args,'adapter_dropout',0.0)

        state = checkpoint_utils.load_checkpoint_to_cpu(self.adapter_model_path)
        #self.w2v_args = Wav2Vec2Config(state["cfg"]["model"]["w2v_args"]["model"])
        if state["cfg"].model.w2v_args is None:
            args_state = checkpoint_utils.load_checkpoint_to_cpu(state["cfg"].model.w2v_path, arg_overrides={})
            #args_state = checkpoint_utils.load_checkpoint_to_cpu("/mnt/zhangyuhao/pretrain/wav2vec_vox_new.pt", arg_overrides={})
            #args_state = checkpoint_utils.load_checkpoint_to_cpu("/mnt/zhangyuhao/pretrain/wav2vec_small.pt", arg_overrides={})
            w2v_args = args_state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(arg_state["args"])
        else:
            w2v_args = state["cfg"].model.w2v_args.model
        self.w2v_args = w2v_args 
        assert self.w2v_args, "the wav2vec args is none!"
        #self.w2v_args.mask_prob=args.mask_prob
        #self.w2v_args.mask_channel_prob=args.mask_channel_prob
        #if self.w2v_args.get("final_dim", None):
        #    self.w2v_args.final_dim=256
        #    self.w2v_args.quantize_targets= True
        self.use_w2v_ctc = getattr(args, "use_w2v_ctc", False)
        from omegaconf import open_dict
        with open_dict(self.w2v_args.model):
            self.w2v_args.model.drop_last_layer = getattr(args, "drop_w2v_last_layer", False)


        if self.use_w2v_ctc:
            self.w2v_ctc_projection = nn.Linear(self.w2v_args.model.encoder_embed_dim, len(task.source_dictionary), bias=False)
            nn.init.normal_(
                self.w2v_ctc_projection.weight, mean=0, std=self.w2v_args.model.encoder_embed_dim ** -0.5
            )
            self.w2v_ctc_dropout_module = FairseqDropout(
                p=args.dropout, module_name=self.__class__.__name__
            )
            self.w2v_softmax = nn.Softmax(dim=-1)
        

        #self.w2v_args = ckpt["args"]
        #self.w2v_args = cfg
        #if "hubert" in state["cfg"].model.w2v_path:
        #    self.wav2vec_model = HubertModel.build_model(self.w2v_args)
        #else:
        #    self.wav2vec_model = Wav2Vec2Model.build_model(self.w2v_args)
        self.w2v_args.task.data = args.data
        self.w2v_args.task.label_dir = args.data
        w2v_task = tasks.setup_task(w2v_args.task)
        self.wav2vec_model = w2v_task.build_model(self.w2v_args.model, from_checkpoint=True)
        self.wav2vec_model.remove_pretraining_modules()

        self.noad=NormalAdapter(state["cfg"].model,task.source_dictionary.pad())
        self.sead=SemanticAdapter(state["cfg"].model,task.source_dictionary.pad())
        if self.w2v_args.model.encoder_embed_dim != state["cfg"].model.encoder_embed_dim:
            d = self.w2v_args.model.encoder_embed_dim
            self.compress_ffn = nn.Linear(d, state["cfg"].model.encoder_embed_dim, True)
        else:
            self.compress_ffn = None
        self.adapter_dropout_module = FairseqDropout(
            p=args.adapter_dropout, module_name=self.__class__.__name__
        )
        noad_state={}
        sead_state={}
        compress_ffn_state={}
        #update the name of parameters to match the wav2vec encoder
        for key in list(state['model'].keys()):
            w = state['model'].pop(key)
            if key.startswith('w2v_encoder'):
                if key.startswith('w2v_encoder.w2v_model.'):
                    new_key = key.replace('w2v_encoder.w2v_model.', '')
                    state['model'][new_key] = w
                if key.startswith('w2v_encoder.noad.'):
                    new_key = key.replace('w2v_encoder.noad.', '')
                    noad_state[new_key] = w
                if key.startswith('w2v_encoder.sead.'):
                    new_key = key.replace('w2v_encoder.sead.', '')
                    sead_state[new_key] = w
                if key.startswith("w2v_encoder.compress_ffn"):
                    new_key = key.replace('w2v_encoder.compress_ffn.', '')
                    compress_ffn_state[new_key] = w
                if self.use_w2v_ctc and key.startswith("w2v_encoder.proj"):
                    self.w2v_ctc_projection.weight.data = w
                    logging.info("load pretrain ctc proj from pretrain wav2vec model")
                    
        self.wav2vec_model.load_state_dict(state["model"],True)
        self.noad.load_state_dict(noad_state,True)
        self.sead.load_state_dict(sead_state,True)
        if self.compress_ffn:
            self.compress_ffn.load_state_dict(compress_ffn_state,True)

        logging.info("load wav2vec2 model: {}".format(self.adapter_model_path))
        
        if self.additional_adapter:
            self.s_adapter=StableAdapter(state["cfg"].model.encoder_embed_dim)

        del self.layers, self.subsample 
        
        #    self.layer_norm = None

    def _get_w2v_feature(self, src_tokens, src_lengths):
        """
        :param src_tokens: b x frames
        :param src_lengths: b-dim length
        :return: w2v_feature: b x short_frames x feature-dim;
                w2v_lengths: b-dim tensor
                w2v_padding_mask: b x short_frames x feature-dim T/F tensor
        """
        padding_mask = lengths_to_padding_mask(src_lengths)
        # print("padding mask:", padding_mask.size())
        # print(padding_mask)
        # w2v_feature = self.wav2vec_model.feature_extractor(src_tokens).transpose(1,2)
        
        w2v_feature, padding_mask = self.wav2vec_model.extract_features(src_tokens, padding_mask)#self.apply_mask and self.training)
        # print("after extraction, padding:", padding_mask)
        output_length = (1 - padding_mask.int()).sum(dim=1)
        # output_length = (torch.ones(padding_mask.size()) - padding_mask.int()).sum(dim=1)

        return w2v_feature, padding_mask, output_length

    def forward(self, src_tokens, src_lengths, **extra_args):
        """
        :param src_tokens: b x frames
        :param src_lengths: b-dim
        """
        # 1. wav2vec
        w2v_feature, encoder_padding_mask, input_lengths = self._get_w2v_feature(
            src_tokens, src_lengths)

        if self.compress_ffn:
            w2v_feature = self.compress_ffn(w2v_feature)
        #adapter for mbart
        if self.noad:
             adapter_feature, encoder_padding_mask = self.noad(w2v_feature, encoder_padding_mask)
        

        # 2. conv-layers
        #print("after w2v extract", w2v_feature.size())
        #conv_feature, input_lengths = self.subsample(w2v_feature, input_lengths)
        # x, input_lengths = self.subsample(src_tokens, src_lengths)
        #print("after conv-3", conv_feature.size())
        #x = self.embed_scale * adapter_feature
        #encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        
        #positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        
        #x += positions
        x = self.dropout_module(adapter_feature)

        if self.sead:
            x, encoder_padding_mask = self.sead(x, encoder_padding_mask)
            x = self.adapter_dropout_module(x)
        if self.additional_adapter:
            x = self.s_adapter(x)
        # 3. layers
        
        #for index,layer in enumerate(self.layers):
        #    x = layer(x, encoder_padding_mask)
        #    if index == 6:
        #        wav_feature=x

        #if not encoder_padding_mask.any():
        #    encoder_padding_mask = None
        
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        #print("encoder output", x.size())
        return {
            "encoder_out": [x],
            "encoder_padding_mask": [encoder_padding_mask],
            "encoder_embedding": [],
            "encoder_states": [],
            "src_tokens": [],
            "src_lengths": [],
            "wav_feature": [adapter_feature]
        }


    def compute_w2v_ctc_logit(self, encoder_out):
        assert self.use_w2v_ctc, "CTC is not available!"

        assert isinstance(encoder_out, dict) and "wav_feature" in encoder_out , "can not get wav feature!"
        w2v_state = encoder_out["wav_feature"][0]
        ctc_logit = self.w2v_ctc_projection(self.w2v_ctc_dropout_module(w2v_state))

        return ctc_logit

    def compute_w2v_ctc_prob(self, encoder_out, temperature=1.0):
        assert self.use_w2v_ctc, "CTC is not available!"

        ctc_logit = self.compute_w2v_ctc_logit(encoder_out) / temperature

        return self.w2v_softmax(ctc_logit)

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """
        new_encoder_out = (
            [] if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            [] if len(encoder_out["encoder_padding_mask"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_padding_mask"]]
        )

        new_encoder_embedding = (
            [] if len(encoder_out["encoder_embedding"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }

    def max_positions(self):
        return self.max_source_positions


@register_model_architecture(model_name="s2t_w2v2_mabrt_sead",
                             arch_name="s2t_w2v2_mabrt_sead")
def base_architecture(args):

    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # conformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
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


@register_model_architecture("s2t_w2v2_mabrt_sead", "s2t_w2v2_mabrt_sead_s")
def s2t_w2v2_mabrt_sead_s(args):
    args.use_asr_finetune_w2v = getattr(args, "use_asr_finetune_w2v", False)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)


@register_model_architecture("s2t_w2v2_mabrt_sead", "s2t_w2v2_mabrt_sead_sp")
def s2t_conformer_w2v2_sp(args):
    args.use_asr_finetune_w2v = getattr(args, "use_asr_finetune_w2v", False)
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_conformer_w2v2_s(args)


@register_model_architecture("s2t_w2v2_mabrt_sead", "s2t_w2v2asr_mabrt_sead_s")
def s2t_w2v2asr_mabrt_sead_s(args):
    args.use_asr_finetune_w2v = getattr(args, "use_asr_finetune_w2v", True)
    s2t_conformer_w2v2_s(args)

