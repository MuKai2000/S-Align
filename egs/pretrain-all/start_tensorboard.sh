#! /bin/bash



exp_name=ende_shrink_AT_baseline
# exp_name=ende_shrink_AT_tokenl
# exp_name=ende_shrink_AT_valloss
#exp_name=ende_v3_merge_wmt_0902_shrink_soft_noCL_AT_sentence_mixup_changeid_scale3.5_alpha1.5_mt0.5
exp_name=ende_v3_merge_wmt_0901_shrink_soft_noCL_AT_sentence_scale3.5_alpha1.5_mt0.5


dataset=mustc
model_dir=./checkpoints/$dataset/st/${exp_name}

tensorboard --logdir=$model_dir