import csv
import io
import logging
import os.path as op
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from fairseq.data import (
    ConcatDataset,
    Dictionary,
    FairseqDataset,
    ResamplingDataset,
    data_utils as fairseq_data_utils,
)
from fairseq.data.audio.audio_utils import get_fbank, get_waveform
from fairseq.data.audio.feature_transforms import CompositeAudioFeatureTransform
from fairseq.data.audio.speech_to_text_dataset import S2TDataConfig, get_features_or_waveform


logger = logging.getLogger(__name__)

def _collate_frames(
    frames: List[torch.Tensor], is_audio_input: bool = False
) -> torch.Tensor:
    """
    Convert a list of 2D frames into a padded 3D tensor
    Args:
        frames (list): list of 2D frames of size L[i]*f_dim. Where L[i] is
            length of i-th frame and f_dim is static dimension of features
    Returns:
        3D tensor of size len(frames)*len_max*f_dim where len_max is max of L[i]
    """
    max_len = max(frame.size(0) for frame in frames)
    if is_audio_input:
        out = frames[0].new_zeros((len(frames), max_len))
    else:
        out = frames[0].new_zeros((len(frames), max_len, frames[0].size(1)))
    for i, v in enumerate(frames):
        out[i, : v.size(0)] = v
    return out

class S2TTripleDataConfig(S2TDataConfig):
    def __init__(self,yaml_path):
        super().__init__(Path(yaml_path))
      
    @property
    def cluster_dict(self):
        """fairseq vocabulary file under data root"""
        return self.config.get("cluster_dict", None)

    @property
    def add_blank_noise(self):
        """add blank noise in text"""
        return self.config.get("add_blank_noise", False)

class TripleDataset(FairseqDataset):
    LANG_TAG_TEMPLATE = "<lang:{}>"

    def __init__(
        self,
        split: str,
        is_train_split: bool,
        data_cfg: S2TTripleDataConfig,
        audio_paths: Optional[List[str]] = None,
        n_frames: Optional[List[int]] = None,
        src_texts: Optional[List[str]] = None,
        tgt_texts: Optional[List[str]] = None,
        speakers: Optional[List[str]] = None,
        src_langs: Optional[List[str]] = None,
        tgt_langs: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        src_dict: Optional[Dictionary] = None,
        tgt_dict: Optional[Dictionary] = None,
        pre_tokenizer=None,
        bpe_tokenizer=None,
        src_bpe_tokenizer=None,
        cluster_convert=None,
        data_type="st"
    ):
        self.split, self.is_train_split = split, is_train_split
        self.data_cfg = data_cfg
        self.audio_paths, self.n_frames = audio_paths, n_frames
        self.data_type = data_type
        if data_type != "mt":
            self.n_samples = len(audio_paths)
        else:
            self.n_samples = len(src_langs)
        if data_cfg.share_src_and_tgt:
            src_texts = tgt_texts
        if data_type != "mt":
            assert len(n_frames) == self.n_samples > 0
        self.add_blank_noise = data_cfg.add_blank_noise 
        assert src_texts is None or len(src_texts) == self.n_samples
        assert tgt_texts is None or len(tgt_texts) == self.n_samples
        assert speakers is None or len(speakers) == self.n_samples
        assert src_langs is None or len(src_langs) == self.n_samples
        assert tgt_langs is None or len(tgt_langs) == self.n_samples
        assert ids is None or len(ids) == self.n_samples
        assert (tgt_dict is None and tgt_texts is None) or (
            tgt_dict is not None and tgt_texts is not None
        )
        self.src_texts, self.tgt_texts = src_texts, tgt_texts
        self.src_langs, self.tgt_langs = src_langs, tgt_langs
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.cluster_convert = cluster_convert
        self.check_tgt_lang_tag()
        self.ids = ids
        self.shuffle = data_cfg.shuffle if is_train_split else False

        self.feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
            self.data_cfg.get_feature_transforms(split, is_train_split)
        )

        self.pre_tokenizer = pre_tokenizer
        self.bpe_tokenizer = bpe_tokenizer
        self.src_bpe_tokenizer = src_bpe_tokenizer

        logger.info(self.__repr__())

    def __repr__(self):
        return (
            self.__class__.__name__
            + f'(split="{self.split}", n_samples={self.n_samples}, '
            f"prepend_tgt_lang_tag={self.data_cfg.prepend_tgt_lang_tag}, "
            f"shuffle={self.shuffle}, transforms={self.feature_transforms})"
        )

    @classmethod
    def is_lang_tag(cls, token):
        pattern = cls.LANG_TAG_TEMPLATE.replace("{}", "(.*)")
        return re.match(pattern, token)

    def check_tgt_lang_tag(self):
        if self.data_cfg.prepend_tgt_lang_tag:
            assert self.tgt_langs is not None and self.tgt_dict is not None
            tgt_lang_tags = [
                self.LANG_TAG_TEMPLATE.format(t) for t in set(self.tgt_langs)
            ]
            assert all(t in self.tgt_dict for t in tgt_lang_tags)

    def tokenize_text(self, text: str, is_src=False):
        if self.pre_tokenizer is not None:
            text = self.pre_tokenizer.encode(text)
        if self.bpe_tokenizer is not None:
            text = self.bpe_tokenizer.encode(text) if not is_src else self.src_bpe_tokenizer.encode(text)
        return text

    def add_blank_noise_in_text(self, line: str):
        org_line = line     # note kkq change here 08.11.2023
        ratio = 0.0
        rm_punk=",.;!?:"
        line = line.replace("...","<s>")
        line = line.replace("..","<s>")
        for w in rm_punk:
            line = line.replace(w,"<s>")
        line=line.replace("  "," ")
        #out.write(line)
        if np.random.random() < 0.2:
            tags = line.strip().split(" ")
            ratio = np.random.random()*0.1+0.1
            length = len(tags)
            add_num = round(len(tags) * ratio)
            for i in range(add_num):
                pos = int(np.random.random() * length)
                while "▁" not in tags[pos]:
                    if pos == length-1:
                        break
                    else:
                        pos+=1
                tags.insert(pos,'<s>')
            line=" ".join(tags)
        return org_line, line, ratio        # note kkq change here 08.11.2023

    def __getitem__(
        self, index: int
    ) -> Tuple[int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        transcript_add_noise = None     # note kkq change here 08.11.2023
        ratio = 0.0
        if self.data_type != "mt":
            source= get_features_or_waveform(
                self.audio_paths[index], need_waveform=self.data_cfg.use_audio_input
            )
            if self.feature_transforms is not None:
                assert not self.data_cfg.use_audio_input
                source = self.feature_transforms(source)
            source = torch.from_numpy(source).float()

            transcript = None
            if self.src_dict is not None and self.src_texts is not None and self.src_bpe_tokenizer is not None:
                tokenized = self.tokenize_text(self.src_texts[index], True)
                if self.add_blank_noise:            # note kkq change here 08.11.2023
                    tokenized, new_tokenized, ratio = self.add_blank_noise_in_text(tokenized)
                    transcript_add_noise = self.src_dict.encode_line(
                        new_tokenized, add_if_not_exist=False, append_eos=True
                    ).long()
                transcript = self.src_dict.encode_line(
                    tokenized, add_if_not_exist=False, append_eos=True
                ).long()
        else:
            if self.src_dict is not None and self.src_texts is not None and self.src_bpe_tokenizer is not None:
                tokenized = self.tokenize_text(self.src_texts[index], True)
                if self.add_blank_noise:
                    _, new_tokenized, ratio = self.add_blank_noise_in_text(tokenized) 
                    tokenized = new_tokenized
                source = self.src_dict.encode_line(
                    tokenized, add_if_not_exist=False, append_eos=True
                ).long()
            transcript = None

        target = None
        if self.tgt_texts is not None:
            tokenized = self.tokenize_text(self.tgt_texts[index])
            target = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True
            ).long()
            if self.data_cfg.prepend_tgt_lang_tag:
                lang_tag = self.LANG_TAG_TEMPLATE.format(self.tgt_langs[index])
                lang_tag_idx = self.tgt_dict.index(lang_tag)
                target = torch.cat((torch.LongTensor([lang_tag_idx]), target), 0)

        return index, source, target, transcript, transcript_add_noise, ratio   # note kkq change here 08.11.2023

    def __len__(self):
        return self.n_samples

    def collater(self, samples: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([i for i, _, _, _, _, _ in samples], dtype=torch.long)
        #if self.data_type != "mt":
        #    frames = _collate_frames(
        #        [s for _, s, _, _ in samples], self.data_cfg.use_audio_input
        #    )
        #else:
        #    frames = fairseq_data_utils.collate_tokens(
        #        [s for _, s, _, _ in samples],
        #        self.tgt_dict.pad(),
        #        self.tgt_dict.eos(),
        #        left_pad=False,
        #        move_eos_to_beginning=False,
        #    )
        frames = _collate_frames(
            [s for _, s, _, _, _, _ in samples], self.data_cfg.use_audio_input
        )
        # sort samples by descending number of frames
        n_frames = torch.tensor([s.size(0) for _, s, _, _, _, _ in samples], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        target, target_lengths = None, None
        prev_output_tokens = None
        ntokens = None
        if self.tgt_texts is not None:
            target = fairseq_data_utils.collate_tokens(
                [t for _, _, t, _, _, _ in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            target = target.index_select(0, order)
            target_lengths = torch.tensor(
                [t.size(0) for _, _, t, _, _, _ in samples], dtype=torch.long
            ).index_select(0, order)
            prev_output_tokens = fairseq_data_utils.collate_tokens(
                [t for _, _, t, _, _, _ in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, order)
            ntokens = sum(t.size(0) for _, _, t, _, _, _ in samples)
        if self.src_dict is not None and self.src_texts is not None and self.data_type != "mt":
            transcript = fairseq_data_utils.collate_tokens(
                [t for _, _, _, t, _, _ in samples],
                self.src_dict.pad(),
                self.src_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            transcript = transcript.index_select(0, order)

            if self.add_blank_noise:
                # note kkq change here 08.11.2023
                transcript_add_noise = fairseq_data_utils.collate_tokens(
                    [t for _, _, _, _, t, _ in samples],
                    self.src_dict.pad(),
                    self.src_dict.eos(),
                    left_pad=False,
                    move_eos_to_beginning=False,
                )
                transcript_add_noise = transcript_add_noise.index_select(0, order)
                transcript_add_noise_lengths = torch.tensor(
                    [t.size(0) for _, _, _, _, t, _ in samples], dtype=torch.long
                ).index_select(0, order)

                ids = torch.tensor(
                    [(1.0-r) for _, _, _, _, _, r in samples], dtype=torch.float
                ).index_select(0, order)
            else:
                transcript_add_noise =None
                transcript_add_noise_lengths = None
                ids = None

            cluster = None
            if self.cluster_convert is not None: 
                cluster_list = []
                for batch in transcript.tolist():
                    tensor = []
                    for token in batch:
                        #if token >3:
                        #    tensor.append(self.cluster_convert.get(token))
                        #else:
                        #    tensor.append(token)
                        tensor.append(self.cluster_convert.get(token, self.cluster_convert.get(3)))
                    cluster_list.append(tensor)
                cluster = torch.Tensor(cluster_list).long()
            #input()
            transcript_lengths = torch.tensor(
                [t.size(0) for _, _, _, t, _, _ in samples], dtype=torch.long
            ).index_select(0, order)
            transcript_ntokens = sum(t.size(0) for _, _, _, t, _, _ in samples)
        else:
            transcript = None
            transcript_lengths = None
            cluster = None
            transcript_ntokens = None
            transcript_add_noise =None
            transcript_add_noise_lengths = None
            ids = None
        out = {
            "id": indices,
            "net_input": {
                "src_tokens": frames,
                "src_lengths": n_frames,
                "prev_output_tokens": prev_output_tokens,
            },
            "transcript": {
                "tokens": transcript,
                "cluster_tokens": cluster,
                "lengths": transcript_lengths,
                "ntokens": transcript_ntokens,
                "tokens_add_noise": transcript_add_noise,
                "lengths_add_noise": transcript_add_noise_lengths,
                "ids": ids,
            },
            "target": target,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
        }
        return out

    def num_tokens(self, index):
        return self.n_frames[index]

    def size(self, index):
        t_len = 0
        if self.tgt_texts is not None:
            tokenized = self.tokenize_text(self.tgt_texts[index])
            t_len = len(tokenized.split(" "))
        return self.n_frames[index], t_len

    @property
    def sizes(self):
        return np.array(self.n_frames)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        # first by descending order of # of frames then by original/random order
        order.append([-n for n in self.n_frames])
        return np.lexsort(order)

    def prefetch(self, indices):
        raise False



class TripleDatasetCreator(object):
    # mandatory columns
    KEY_ID, KEY_AUDIO, KEY_N_FRAMES = "id", "audio", "n_frames"
    KEY_TGT_TEXT = "tgt_text"
    # optional columns
    KEY_SPEAKER, KEY_SRC_TEXT = "speaker", "src_text"
    KEY_SRC_LANG, KEY_TGT_LANG = "src_lang", "tgt_lang"
    # default values
    DEFAULT_ID = DEFAULT_AUDIO = DEFAULT_N_FRAMES = DEFAULT_SPEAKER = DEFAULT_SRC_TEXT = DEFAULT_LANG = ""

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[List[Dict]],
        data_cfg: S2TTripleDataConfig,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
        src_dict=None,
        src_bpe_tokenizer=None,
        cluster_convert=None,
        data_type="st"
    ) -> TripleDataset:
        audio_paths, n_frames, src_texts, tgt_texts, ids = [], [], [], [], []
        speakers, src_langs, tgt_langs = [], [], []
        for s in samples:
            if data_type != "mt":
                ids.extend([ss[cls.KEY_ID] for ss in s])
                audio_paths.extend(
                    [op.join(data_cfg.audio_root, ss[cls.KEY_AUDIO]) for ss in s]
                )
                n_frames.extend([int(ss[cls.KEY_N_FRAMES]) for ss in s])
            else:
                ids.extend([ss.get(cls.KEY_ID, cls.DEFAULT_ID) for ss in s])
                audio_paths.extend([ss.get(cls.DEFAULT_AUDIO) for ss in s])
                #n_frames.extend([ss.get(cls.KEY_N_FRAMES, cls.DEFAULT_N_FRAMES) for ss in s])
                n_frames.extend([len(ss[cls.KEY_SRC_TEXT].split(" ")) for ss in s])
            tgt_texts.extend([ss[cls.KEY_TGT_TEXT] for ss in s])
            src_texts.extend(
                [ss.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for ss in s]
            )
            speakers.extend([ss.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for ss in s])
            src_langs.extend([ss.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for ss in s])
            tgt_langs.extend([ss.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for ss in s])

        return TripleDataset(
            split_name,
            is_train_split,
            data_cfg,
            audio_paths,
            n_frames,
            src_texts,
            tgt_texts,
            speakers,
            src_langs,
            tgt_langs,
            ids,
            src_dict,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            src_bpe_tokenizer,
            cluster_convert,
            data_type
        )

    @classmethod
    def _get_size_ratios(cls, ids: List[str], sizes: List[int], alpha: float = 1.0):
        """Size ratios for temperature-based sampling
        (https://arxiv.org/abs/1907.05019)"""
        _sizes = np.array(sizes)
        prob = _sizes / _sizes.sum()
        smoothed_prob = prob ** alpha
        smoothed_prob = smoothed_prob / smoothed_prob.sum()
        size_ratio = (smoothed_prob * _sizes.sum()) / _sizes

        o_str = str({_i: f"{prob[i]:.3f}" for i, _i in enumerate(ids)})
        logger.info(f"original sampling probability: {o_str}")
        p_str = str({_i: f"{smoothed_prob[i]:.3f}" for i, _i in enumerate(ids)})
        logger.info(f"balanced sampling probability: {p_str}")
        sr_str = str({_id: f"{size_ratio[i]:.3f}" for i, _id in enumerate(ids)})
        logger.info(f"balanced sampling size ratio: {sr_str}")
        return size_ratio.tolist()

    @classmethod
    def from_tsv(
        cls,
        root: str,
        data_cfg: S2TTripleDataConfig,
        splits: str,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
        is_train_split: bool,
        epoch: int,
        seed: int,
        src_dict=None,
        src_bpe_tokenizer=None,
        cluster_convert=None,
        data_type="st"
    ) -> TripleDataset:
        samples = []
        _splits = splits.split(",")
        assert data_type in ["st","mt","asr"]
        for split in _splits:
            tsv_path = op.join(root, f"{split}.tsv")
            if not op.isfile(tsv_path):
                raise FileNotFoundError(f"Dataset not found: {tsv_path}")
            with open(tsv_path) as f:
                reader = csv.DictReader(
                    f,
                    delimiter="\t",
                    quotechar=None,
                    doublequote=False,
                    lineterminator="\n",
                    quoting=csv.QUOTE_NONE,
                )
                samples.append([dict(e) for e in reader])
                assert len(samples) > 0

        datasets = [
            cls._from_list(
                name,
                is_train_split,
                [s],
                data_cfg,
                tgt_dict,
                pre_tokenizer,
                bpe_tokenizer,
                src_dict=src_dict,
                src_bpe_tokenizer=src_bpe_tokenizer,
                cluster_convert=cluster_convert,
                data_type=data_type
            )
            for name, s in zip(_splits, samples)
        ]

        if is_train_split and len(_splits) > 1 and data_cfg.sampling_alpha != 1.0:
            # temperature-based sampling
            size_ratios = cls._get_size_ratios(
                _splits, [len(s) for s in samples], alpha=data_cfg.sampling_alpha
            )
            datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=seed, epoch=epoch, replace=(r >= 1.0)
                )
                for d, r in zip(datasets, size_ratios)
            ]
        return ConcatDataset(datasets)
