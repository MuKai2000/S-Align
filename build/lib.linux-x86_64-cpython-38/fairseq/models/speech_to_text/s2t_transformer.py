# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.speech_to_text.modules.convolution import (
    Conv1dSubsampler,
    Conv2dSubsampler,
)
from fairseq.models.transformer import Embedding, TransformerDecoder
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
    CreateLayerHistory,
)
from torch import Tensor

logger = logging.getLogger(__name__)


class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)


@register_model("s2t_transformer")
class S2TTransformerModel(FairseqEncoderDecoderModel):
    """Adapted Transformer model (https://arxiv.org/abs/1706.03762) for
    speech-to-text tasks. The Transformer encoder/decoder remains the same.
    A trainable input subsampler is prepended to the Transformer encoder to
    project inputs into the encoder dimension as well as downsample input
    sequence for computational efficiency."""

    @classmethod
    def hub_models(cls):
        base_url = "http://dl.fbaipublicfiles.com/fairseq/s2t"
        model_ids = [
            "s2t_transformer_s-en-asr-librispeech",
            "s2t_transformer_m-en-asr-librispeech",
            "s2t_transformer_l-en-asr-librispeech",
        ]
        return {i: f"{base_url}/{i}.tar.gz" for i in model_ids}

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        config_yaml="config.yaml",
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            config_yaml=config_yaml,
            **kwargs,
        )
        return S2THubInterface(x["args"], x["task"], x["models"][0])

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # input
        parser.add_argument(
            "--conv-kernel-sizes",
            type=str,
            metavar="STR",
            help="kernel sizes of Conv1d (s2t_transformer) subsampling layers",
        )
        parser.add_argument(
            "--conv-channels",
            type=int,
            metavar="N",
            help="# of channels in Conv1d (s2t_transformer) subsampling layers",
        )
        parser.add_argument(
            "--conv-out-channels",
            type=int,
            metavar="N",
            help="# of channels in Conv2d (convtransformer) subsampling layers",
        )
        parser.add_argument(
            "--conv-version",
            type=str,
            default="s2t_transformer",
            choices=["s2t_transformer", "convtransformer"],
            help="version of frontend convolutional layers",
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
            "--encoder-attention-type",
            type=str,
            default="selfattn",
            choices=[
                "selfattn",
                "rel_selfattn",
                "relative",
            ],
            help="transformer encoder self-attention layer type"
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
            "--decoder-attention-type",
            type=str,
            default="selfattn",
            choices=[
                "selfattn",
                "rel_selfattn",
                "relative",
            ],
            help="transformer decoder self-attention layer type"
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
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--share-all-embeddings',
                            action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
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
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--max-encoder-relative-length', type=int, default=-1,
                            help='the max relative length')
        parser.add_argument('--max-decoder-relative-length', type=int, default=-1,
                            help='the max relative length')
        parser.add_argument('--k-only', default=False, action='store_true',
                            help='select the relative mode to map relative position information')
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--load-pretrained-decoder-from",
            type=str,
            metavar="STR",
            help="model to take decoder weights from (for initialization)",
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
            "--use-enc-dlcl",
            default=False,
            action='store_true',
            help="use dlcl encoder",
        )
        parser.add_argument(
            "--use-dec-dlcl",
            default=False,
            action='store_true',
            help="use dlcl encoder",
        )
        parser.add_argument(
            '--encoder-history-type',
            default="learnable_dense",
            help='encoder layer history type'
        )
        parser.add_argument(
            '--decoder-history-type',
            default="learnable_dense",
            help='decoder layer history type'
        )
        pass

    @classmethod
    def build_encoder(cls, args, task=None, embed_tokens=None):
        encoder = S2TTransformerEncoder(args, task, embed_tokens)
        if getattr(args, "load_pretrained_encoder_from", None):
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
            print("__________load pretrained encoder____________")
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from, strict=False
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
            print("__________load pretrained decoder____________")
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from, strict=False
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
        #print(task.target_dictionary.symbols)
        #print(task.source_dictionary.symbols)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim, getattr(args, "decoder_embed_path", None)
        )
        args.tgt_dict_size = len(task.target_dictionary)
        encoder = cls.build_encoder(args, task, decoder_embed_tokens)
        if getattr(args, "encoder_freeze_module", None):
            utils.freeze_parameters(encoder, args.encoder_freeze_module)
            logging.info("freeze the encoder module: {}".format(args.encoder_freeze_module))
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        if getattr(args, "decoder_freeze_module", None):
            utils.freeze_parameters(decoder, args.decoder_freeze_module)
            logging.info("freeze the decoder module: {}".format(args.decoder_freeze_module))
        if hasattr(args, 'tune_mbart_LNA'):
            #utils.freeze_parameters(decoder, "decoder")
            for key, value in decoder.layers.named_parameters():
               if "layer_norm" in key:
                   value.requires_grad = True
               elif "encoder_attn" in key:
                   value.requires_grad = True
               else:
                   value.requires_grad = False
            logging.info("only tune the decoder layer normal and encoder-attn module")
        if hasattr(args, 'tune_encoder_LNA'):
            #utils.freeze_parameters(encoder.layers, "layers")
            for key, value in encoder.layers.named_parameters():
               if "layer_norm" in key:
                   value.requires_grad = True
               #elif "self_attn" in key:
               #    value.requires_grad = True
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

    def get_ctc_target(self, sample: Optional[Dict[str, Tensor]]):
        return sample["target"], sample["target_lengths"]

    def get_ctc_output(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        sample: Optional[Dict[str, Tensor]],
    ):
        encoder_out = net_output[1]["encoder_out"]["encoder_out"][0]
        logits = self.encoder.ctc_proj(encoder_out)  # T x B x C
        out = utils.log_softmax(logits.float(), dim=-1)
        padding_mask = net_output[1]["encoder_out"]["encoder_padding_mask"]
        lens = out.new_full((out.shape[1],), out.shape[0]).long()
        if len(padding_mask) > 0:
            lens -= padding_mask[0].sum(dim=-1)
        return out, lens

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        return decoder_out


class S2TTransformerEncoder(FairseqEncoder):
    """Speech-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args, task=None, embed_tokens=None):
        super().__init__(None)

        #self.encoder_freezing_updates = args.encoder_freezing_updates
        self.num_updates = 0

        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = 1

        self.conv_version = getattr(args, "conv_version", "s2t_transformer")
        if self.conv_version == "s2t_transformer":
            self.subsample = Conv1dSubsampler(
                args.input_feat_per_channel * args.input_channels,
                args.conv_channels,
                args.encoder_embed_dim,
                [int(k) for k in args.conv_kernel_sizes.split(",")],
            )
        elif self.conv_version == "convtransformer":
            self.subsample = Conv2dSubsampler(
                args.input_channels,
                args.input_feat_per_channel,
                args.conv_out_channels,
                args.encoder_embed_dim,
            )
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, args.encoder_embed_dim, self.padding_idx
        )

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

        if getattr(args, "use_enc_dlcl", False):
            self.history = CreateLayerHistory(args, is_encoder=True)
        else:
            self.history = None

        self.use_ctc = getattr(args, "ctc_weight", 0.0) > 0.0
        if self.use_ctc:
            print(task.source_dictionary , task.target_dictionary)
            if task.source_dictionary == task.target_dictionary and getattr(args, "share_all_embeddings", False):
                self.ctc_projection = nn.Linear(
                    embed_tokens.weight.shape[1],
                    embed_tokens.weight.shape[0],
                    bias=False,
                )
                self.ctc_projection.weight = embed_tokens.weight
            else:
                self.ctc_projection = nn.Linear(args.encoder_embed_dim, len(task.source_dictionary), bias=False)
                nn.init.normal_(
                    self.ctc_projection.weight, mean=0, std=args.encoder_embed_dim ** -0.5
                )
            self.ctc_dropout_module = FairseqDropout(
                p=args.dropout, module_name=self.__class__.__name__
            )
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, src_tokens, src_lengths):

        if self.history is not None:
            self.history.clean()
        x, input_lengths = self.subsample(src_tokens, src_lengths)
        x = self.embed_scale * x

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        x += positions
        x = self.dropout_module(x)
        positions = self.dropout_module(positions)

        # add emb into history
        if self.history is not None:
            self.history.add(x)

        for layer in self.layers:
            if self.history is not None:
                x = self.history.pop()
            x = layer(x, encoder_padding_mask, pos_emb=positions)
            if self.history is not None:
                self.history.add(x)

        if self.history is not None:
            x = self.history.pop()

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }


    def compute_ctc_logit(self, encoder_out):
        assert self.use_ctc, "CTC is not available!"

        if isinstance(encoder_out, dict) and "encoder_out" in encoder_out:
            encoder_state = encoder_out["encoder_out"][0]
        else:
            encoder_state = encoder_out
        ctc_logit = self.ctc_projection(self.ctc_dropout_module(encoder_state))

        return ctc_logit

    def compute_ctc_prob(self, encoder_out, temperature=1.0):
        assert self.use_ctc, "CTC is not available!"

        ctc_logit = self.compute_ctc_logit(encoder_out) / temperature

        return self.softmax(ctc_logit)

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            []
            if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            []
            if len(encoder_out["encoder_padding_mask"]) == 0
            else [
                x.index_select(0, new_order)
                for x in encoder_out["encoder_padding_mask"]
            ]
        )

        new_encoder_embedding = (
            []
            if len(encoder_out["encoder_embedding"]) == 0
            else [
                x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]
            ]
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

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


class TransformerDecoderScriptable(TransformerDecoder):
    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        # call scriptable method from parent class
        x, _ = self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )
        return x, None

    def get_normalized_probs_scriptable(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None:
            if sample is not None:
                assert "target" in sample
                target = sample["target"]
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output[0]
        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)



@register_model_architecture(model_name="s2t_transformer", arch_name="s2t_transformer")
def base_architecture(args):
    #args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    # Convolutional subsampler
    args.input_channels = getattr(args, "input_channels", 1)
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")  # for Conv1d
    args.conv_channels = getattr(args, "conv_channels", 1024)  # for Conv1d
    args.conv_out_channels = getattr(args, "conv_out_channels", 256)  # for Conv2d
    args.conv_version = getattr(args, "conv_version", "s2t_transformer")
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_type = getattr(args, "encoder_attention_type", "selfattn")
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_type = getattr(args, "decoder_attention_type", "selfattn")
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
        args, "share_decoder_input_output_embed", False
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

    args.max_encoder_relative_length = getattr(args, 'max_encoder_relative_length', -1)
    args.max_decoder_relative_length = getattr(args, 'max_decoder_relative_length', -1)
    args.k_only = getattr(args, 'k_only', True)


@register_model_architecture("s2t_transformer", "s2t_transformer_s")
def s2t_transformer_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)


@register_model_architecture("s2t_transformer", "s2t_transformer_s_relative")
def s2t_transformer_s_relative(args):
    args.max_encoder_relative_length = 100
    args.max_decoder_relative_length = 20
    args.k_only = True
    s2t_transformer_s(args)


@register_model_architecture("s2t_transformer", "s2t_transformer_xs")
def s2t_transformer_xs(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 4)
    args.dropout = getattr(args, "dropout", 0.3)
    s2t_transformer_s(args)


@register_model_architecture("s2t_transformer", "s2t_transformer_sp")
def s2t_transformer_sp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_transformer_s(args)


@register_model_architecture("s2t_transformer", "s2t_transformer_m")
def s2t_transformer_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    base_architecture(args)


@register_model_architecture("s2t_transformer", "s2t_transformer_mp")
def s2t_transformer_mp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_transformer_m(args)


@register_model_architecture("s2t_transformer", "s2t_transformer_l")
def s2t_transformer_l(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.2)
    base_architecture(args)


@register_model_architecture("s2t_transformer", "s2t_transformer_lp")
def s2t_transformer_lp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_transformer_l(args)
