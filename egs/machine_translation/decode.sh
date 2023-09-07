#!/usr/bin/bash
set -e
# source /home/zhangyuhao/VENV/AT/bin/activate
model_root_dir=checkpoints
root=/mnt/zhangyh/fairseq-AT/
# set task
#task=mustc
task=wmt-en2fr
#task=wmt-en2es
# task=wmt-en2de
# set tag
#model_dir_tag=mustc-ende-baseline-silent
#model_dir_tag=mustc-ende-iwslt-prenorm-conv2-prepos-silent-rpr
#model_dir_tag=iwslt23-conv-silent-conv5-check
#model_dir_tag=mustc-ende-iwslt-prenorm-conv2-prepos-silent
#model_dir_tag=mustc-ende-iwslt-prenorm-conv2-prepos-silent-2048
#model_dir_tag=iwslt23-conv-silent-encoder3-conv5
#model_dir_tag=iwslt23-conv-silent-encoder-all-conv5-l2g
#model_dir_tag=mustc-ende-baseline
#model_dir_tag=iwslt23-conv-silent-conv5-all-check
#model_dir_tag=mergerwmt-enes
#model_dir_tag=ende-baseline
#model_dir_tag=ende-baseline-mustc-ft
#model_dir_tag=enfr-baseline
#model_dir_tag=enes-baseline
#model_dir_tag=enes-baseline-mustc-ft
model_dir_tag=enfr-baseline-mustc-ft

model_dir=$model_root_dir/$task/$model_dir_tag

# checkpoint=checkpoint_best.pt
#checkpoint=checkpoint22.pt
checkpoint=last8.ensemble.pt

ensemble=8

gpu=0

if [ -n "$ensemble" ]; then
        if [ ! -e "$model_dir/last$ensemble.ensemble.pt" ]; then
                PYTHONPATH=`pwd` python3 /mnt/zhangyh/fairseq-AT/scripts/average_checkpoints.py --inputs $model_dir --output $model_dir/last$ensemble.ensemble.pt --num-epoch-checkpoints $ensemble
        fi
        checkpoint=last$ensemble.ensemble.pt
fi

output=$model_dir/translation.log

if [ -n "$cpu" ]; then
        use_cpu=--cpu
fi

export CUDA_VISIBLE_DEVICES=$gpu

python3 $root/fairseq_cli/generate.py \
data-bin/mustc-enfr-silent \
--path $model_dir/$checkpoint \
--gen-subset test \
--batch-size 32 \
--lenpen 1.0 \
--max-source-positions 500 \
--scoring sacrebleu \
--source-lang en \
--target-lang fr \
--beam 8 \
--remove-bpe sentencepiece \
--results-path ${model_dir} 

tail -n3  $model_dir/generate-test.txt
#--post-process sentencepiece \
#--tokenizer moses \
#--sacrebleu-tokenizer spm \

