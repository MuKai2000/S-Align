#! /bin/bash

# Processing Libri Datasets

# Copyright 2021 Natural Language Processing Laboratory 
# Yuhao Zhang (yoohao.zhang@gmail.com)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',


set -e
#set -u
set -o pipefail
export PYTHONIOENCODING=UTF-8

eval=1
time=$(date "+%m%d_%H%M")

stage=1
stop_stage=1

######## hardware ########
# devices
device=(0,1,2,3,4,5,6,7)
#device=()
gpu_num=8
update_freq=1

root_dir=/workspace/fairseq-0.12.3
pwd_dir=$PWD


# dataset
src_lang=en
tgt_lang=de
lang=${src_lang}-${tgt_lang}

dataset=mustc
task=joint_triple_pretraining
vocab_type=unigram
asr_vocab_size=5000
vocab_size=10000
share_dict=1
speed_perturb=0
lcrm=1
tokenizer=0

use_specific_dict=0
specific_prefix=valid
specific_dir=
asr_vocab_prefix=spm_unigram10000_st_share
st_vocab_prefix=spm_unigram10000_st_share

org_data_dir=raw_data
#data_dir=data_g_no_specaug
data_dir=data_all_ende
#data_dir=data_all_enfr
#data_dir=data_g
#data_dir=data_ende_gpt
test_subset=test

# exp
exp_prefix=${time}
extra_tag=
extra_parameter=
exp_tag=
#exp_name=conformer_ctc_perturb_gtrans_no_specagument
#exp_name=conformer_ctc_perturb_gtrans_pretrain_all_xlmr_kd_embedding_vocab1W

#exp_name=transformer_w2v_perturb_gtrans
#exp_name=st_joint_train_8gpu_ende_cl
#exp_name=st_joint_train_8gpu_ende_word_ctc_shrink
#exp_name=st_joint_train_8gpu_ende_word_ctc_cr_word_align
#exp_name=st_joint_ctc_all_shrink_uniq_word_align_task_relax_ban_lookback_bigbatch
#exp_name=st_joint_ctc_all_shrink_uniq_word_align_task_relax_lookback_bigbatch_two_cl
#exp_name=st_joint_ctc_all_shrink_uniq_word_align_task_relax_lookback_bigbatch_two_cl_conv_sead
#exp_name=st_joint_ctc_all_shrink_uniq_word_align_task_relax_lookback_bigbatch_two_cl_conv_silent_noise
#exp_name=st_joint_ctc_all_shrink_uniq_word_align_task_relax_lookback_bigbatch_two_cl_encoder_conv_noise_new
#exp_name=st_joint_ctc_all_shrink_uniq_word_align_task_relax_lookback_bigbatch_two_cl_encoder3_conv3_noise_banscale_maxmt_weight
#exp_name=st_joint_ctc_all_shrink_uniq_word_align_task_relax_lookback_bigbatch_two_cl_encoder_conv5_dy_l2g_noise_banscale_maxmt_weight_kd
# exp_name=ende_shrink_AT_scale10
exp_name=ende_shrink_AT_val_loss_test
#exp_name=st_joint_ctc_all_shrink_uniq_word_align_task_relax_lookback_bigbatch_two_cl_conv_noise_banscale_maxmt_weight_repeat
#exp_name=st_joint_ctc_shrink_fix_padding
fine_tune=
use_w2v_ctc=1
apply_mask=

# config
#train_config=train_ctc_conformer.yaml
#train_config=train_transformer_w2v.yaml
#train_config=train_joint_hubert_bce.yaml
#train_config=train_joint_hubert_bce.yaml
#train_config=train_joint_hubert_word_ctc_shrink.yaml
#train_config=train_joint_hubert_cluster_task_relax.yaml
train_config=train_shrink_AT.yaml
#train_config=train_joint_hubert_baseline.yaml
#train_config=train_ctc_conformer_pretrain_ft_st.yaml
freeze_decode_module=
#freeze_encode_module=wav2vec_model
share_decoder_input_output_embed=1
#decoder_embed_path=/apdcephfs/share_1157259/users/adrienxu/st/pretrain-text/mbart/embeddings
#decoder_embed_path=/workspace/MSP-ST/fairseq/egs/machine_translation/pretrain_embeddings
#decoder_embed_path=/apdcephfs/share_1157259/users/adrienxu/st/pretrain-text/xlmr.large/embeddings_kd
tune_w2v_LNA=
tune_mbart_LNA=
tune_encoder_LNA=

# training setting
fp16=1
max_tokens=15000
max_batch_size=
#max_tokens=3000
step_valid=1
bleu_valid=1
save_interval_updates=4000
keep_interval_updates=8
save_interval=1
# decoding setting
dec_model=checkpoint_best.pt
n_average=10
beam_size=5
len_penalty=1.2
#load_pretrain_encoder=checkpoints/libri_trans/st/conformer_ctc_perturb_gtrans_pretrain_wav2vec_content_xlmr_ffn_embedding_vocab1W/checkpoint_last.pt
#load_pretrain_decoder=/apdcephfs/share_1157259/users/adrienxu/st/pretrain-text/mbart/mbart.cc25.v2/model.pt
#load_pretrain_encoder=/apdcephfs/share_1157259/users/adrienxu/st/pretrain-text/mbart/mbart.cc25.v2/model.pt
#load_pretrain_decoder=/workspace/fairseq-0.10.2/egs/mt-finetune/checkpoints/mbart-ende/checkpoint_best.pt
#load_pretrain_decoder=/workspace/MSP-ST/fairseq/egs/machine_translation/checkpoints/wmt-en2de/merge-lcrm/last5.ensemble.pt
#load_pretrain_encoder=/apdcephfs/share_1157259/users/adrienxu/st/pretrain-text/mt-finetune/checkpoints/model.pt
#load_encoder_layers_from=/apdcephfs/share_1157259/users/adrienxu/st/pretrain-text/mt-finetune/checkpoints/model.pt
if [[ $fine_tune -eq 1 ]]; then
    train_config=tune_st_hubert_shrink.yaml
    max_tokens=5000
    step_valid=1
    save_interval_updates=100
    exp_name=${exp_name}_tune
    tune_encoder_LNA=1
    tune_w2v_LNA=1
fi

if [[ ${share_dict} -eq 1 ]]; then
	data_config=config_st.yaml
else
	data_config=config_st.yaml
fi
if [[ ${speed_perturb} -eq 1 ]]; then
    data_dir=${data_dir}_sp
    exp_prefix=${exp_prefix}_sp
fi
if [[ ${lcrm} -eq 1 ]]; then
    data_dir=${data_dir}_lcrm
    exp_prefix=${exp_prefix}_lcrm
fi
if [[ ${use_specific_dict} -eq 1 ]]; then
    data_dir=${data_dir}_${specific_prefix}
    exp_prefix=${exp_prefix}_${specific_prefix}
fi
if [[ ${tokenizer} -eq 1 ]]; then
    data_dir=${data_dir}_tok
    exp_prefix=${exp_prefix}_tok
fi

. ./local/parse_options.sh || exit 1;

# full path
train_config=$pwd_dir/conf/${train_config}
if [[ -z ${exp_name} ]]; then
    exp_name=${exp_prefix}_$(basename ${train_config%.*})_${exp_tag}
    if [[ -n ${extra_tag} ]]; then
        exp_name=${exp_name}_${extra_tag}
    fi
fi
model_dir=./checkpoints/$dataset/st/${exp_name}

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    # pass
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
   
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: ASR Data Preparation"
    if [[ ! -e ${data_dir}/${lang} ]]; then
        mkdir -p ${data_dir}/${lang}
    fi

    cmd="python ${root_dir}/examples/speech_to_text/prep_libri_trans_data.py
        --data-root ${org_data_dir}
        --output-root ${data_dir}
        --task asr
        --src-lang en --tgt-lang fr
        --vocab-type ${vocab_type}
        --vocab-size ${asr_vocab_size}"
    if [[ ${speed_perturb} -eq 1 ]]; then
        cmd="$cmd
        --speed-perturb"
    fi
    echo -e "\033[34mRun command: \n${cmd} \033[0m"
    [[ $eval -eq 1 && ${share_dict} -ne 1 && ${use_specific_dict} -ne 1 ]] && eval $cmd
    asr_prefix=spm_${vocab_type}${asr_vocab_size}_asr

    echo "stage 0: ST Data Preparation"
    cmd="python ${root_dir}/examples/speech_to_text/prep_libri_trans_data.py
        --data-root ${org_data_dir}
        --output-root ${data_dir}
        --task st
        --src-lang en --tgt-lang fr
        --add-src
        --cmvn-type utterance
        --vocab-type ${vocab_type}
        --vocab-size ${vocab_size}"

    if [[ ${use_specific_dict} -eq 1 ]]; then
        cp -r ${specific_dir}/${asr_vocab_prefix}.* ${data_dir}/${lang}
        cp -r ${specific_dir}/${st_vocab_prefix}.* ${data_dir}/${lang}
        if [[ $share_dict -eq 1 ]]; then
            cmd="$cmd
        --share
        --st-spm-prefix ${st_vocab_prefix}"
        else
            cmd="$cmd
        --st-spm-prefix ${st_vocab_prefix}
        --asr-prefix ${asr_vocab_prefix}"
        fi
    else
        if [[ $share_dict -eq 1 ]]; then
            cmd="$cmd
        --share"
        else
            cmd="$cmd
        --asr-prefix ${asr_prefix}"
        fi
    fi
    if [[ ${speed_perturb} -eq 1 ]]; then
        cmd="$cmd
        --speed-perturb"
    fi
    if [[ ${lcrm} -eq 1 ]]; then
        cmd="$cmd
        --lowercase-src
        --rm-punc-src"
    fi
    if [[ ${tokenizer} -eq 1 ]]; then
        cmd="$cmd
        --tokenizer"
    fi
    if [[ ${specaugment} -eq 0 ]]; then
        cmd="$cmd
        --specaugment-policy None"
    fi

    echo -e "\033[34mRun command: \n${cmd} \033[0m"
    [[ $eval -eq 1 ]] && eval ${cmd}
    deactivate
fi

data_dir=${data_dir}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: ST Network Training"
    [[ ! -d ${data_dir} ]] && echo "The data dir ${data_dir} is not existing!" && exit 1;

    if [[ -z ${device} || ${#device[@]} -eq 0 ]]; then
		if [[ ${gpu_num} -eq 0 ]]; then
			device=()
		else
        	source ./local/utils.sh
        	device=$(get_devices $gpu_num 0)
		fi
    fi

    echo -e "dev=${device} data=${data_dir} model=${model_dir}"

    if [[ ! -d ${model_dir} ]]; then
        mkdir -p ${model_dir}
    else
        echo "${model_dir} exists."
    fi

    cp ${BASH_SOURCE[0]} ${model_dir}
    cp ${PWD}/joint_train.sh ${model_dir}
    cp ${train_config} ${model_dir}

    cmd="python3 -u ${root_dir}/fairseq_cli/train.py
        ${data_dir}
        --config-yaml ${data_config}
        --train-config ${train_config}
        --task ${task}
        --max-tokens ${max_tokens}
        --skip-invalid-size-inputs-valid-test
        --update-freq ${update_freq}
        --log-interval 100
        --save-dir ${model_dir}
        --tensorboard-logdir ${model_dir}
        "


    if [[ -n ${extra_parameter} ]]; then
        cmd="${cmd}
        ${extra_parameter}"
    fi
	if [[ ${gpu_num} -gt 0 ]]; then
		cmd="${cmd}
        --distributed-world-size $gpu_num
        --ddp-backend no_c10d"
	fi
    if [[ $fp16 -eq 1 ]]; then
        cmd="${cmd}
        --fp16"
    fi
    if [[ $step_valid -eq 1 ]]; then
        validate_interval=1
        save_interval=1
        keep_last_epochs=2
        no_epoch_checkpoints=1
        save_interval_updates=10 # 2000
        keep_interval_updates=5
    else
        validate_interval=1
        keep_last_epochs=10
    fi
    if [[ $bleu_valid -eq 1 ]]; then
        cmd="$cmd
        --eval-bleu
        --eval-tokenized-bleu
        --eval-bleu-remove-bpe sentencepiece
        --best-checkpoint-metric bleu
        --keep-best-checkpoints 10
        --maximize-best-checkpoint-metric"
        #--eval-bleu-args '{\"beam\": 1}'"
    fi
    echo $cmd
    if [[ -n $no_epoch_checkpoints && $no_epoch_checkpoints -eq 1 ]]; then
        cmd="$cmd
        --no-epoch-checkpoints"
    fi
    if [[ -n $validate_interval ]]; then
        cmd="${cmd}
        --validate-interval $validate_interval "
    fi
    if [[ -n $save_interval ]]; then
        cmd="${cmd}
        --save-interval $save_interval "
    fi
    if [[ -n $keep_last_epochs ]]; then
        cmd="${cmd}
        --keep-last-epochs $keep_last_epochs "
    fi
    if [[ -n $save_interval_updates ]]; then
        cmd="${cmd}
        --save-interval-updates $save_interval_updates"
        if [[ -n $keep_interval_updates ]]; then
        cmd="${cmd}
        --keep-interval-updates $keep_interval_updates"
        fi
    fi
    if [[ -n $freeze_decode_module ]]; then
        cmd="${cmd}
        --decoder-freeze-module $freeze_decode_module"
    fi
    if [[ -n $freeze_encode_module ]]; then
        cmd="${cmd}
        --encoder-freeze-module $freeze_encode_module"
    fi
    if [[ -n $decoder_embed_path ]]; then
        cmd="${cmd}
        --decoder-embed-path $decoder_embed_path"
    fi
    if [[ -n $load_pretrain_encoder ]]; then
        cmd="${cmd}
        --load-pretrained-encoder-from $load_pretrain_encoder"
    fi
    if [[ -n $load_pretrain_decoder ]]; then
        cmd="${cmd}
        --load-pretrained-decoder-from $load_pretrain_decoder"
    fi
    if [[ -n $share_decoder_input_output_embed ]]; then
        cmd="${cmd}
        --share-decoder-input-output-embed"
    fi
    if [[ -n $fine_tune ]]; then
        cmd="${cmd}
        --reset-lr-scheduler 
        --reset-dataloader
        --reset-optimizer"
    fi
    if [[ -n $use_w2v_ctc ]]; then
        cmd="${cmd}
        --use-w2v-ctc"
    fi
    if [[ -n $tune_w2v_LNA ]]; then
        cmd="${cmd}
        --tune-w2v-LNA"
    fi
    if [[ -n $tune_mbart_LNA ]]; then
        cmd="${cmd}
        --tune-mbart-LNA"
    fi
    if [[ -n $tune_encoder_LNA ]]; then
        cmd="${cmd}
        --tune-encoder-LNA"
    fi
    if [[ -n $load_encoder_layers_from ]]; then
        cmd="${cmd}
        --load-encoder-layers-from $load_encoder_layers_from"
    fi
    if [[ -n $max_batch_size ]]; then
        cmd="${cmd}
        --batch-size $max_batch_size"
    fi
    if [[ -n $apply_mask ]]; then
        cmd="${cmd}
        --apply-mask --mask-prob 0.5 --mask-channel-prob 0.25"
    fi

    echo -e "\033[34mRun command: \n${cmd} \033[0m"

    # save info
    log=./history.log
    echo "${time} | ${device} | ${data_dir} | ${model_dir} " >> $log
    cat $log | tail -n 50 > tmp.log
    mv tmp.log $log

    echo "export CUDA_VISIBLE_DEVICES=${device}"

    #cmd="nohup ${cmd} >> ${model_dir}/train.log 2>&1 &"
    #CUDA_VISIBLE_DEVICES=${device} ${cmd}
    cmd="CUDA_VISIBLE_DEVICES=${device} nohup ${cmd} >> ${model_dir}/train.log 2>&1 &"
    if [[ $eval -eq 1 ]]; then
        eval $cmd
        sleep 2s
        tail -f  ${model_dir}/train.log
    		tail -n `wc -l ${model_dir}/train.log | awk '{print $1+1}'` -f ${model_dir}/train.log
    fi
fi
wait

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: ST Decoding"
    if [[ ${n_average} -ne 1 ]]; then
        # Average models
		dec_model=avg_${n_average}_checkpoint.pt

		cmd="python ${root_dir}/scripts/average_checkpoints.py
        --inputs ${model_dir}
        --num-epoch-checkpoints ${n_average}
        --output ${model_dir}/${dec_model}"
    	echo -e "\033[34mRun command: \n${cmd} \033[0m"
    	[[ $eval -eq 1 ]] && eval $cmd
	else
		dec_model=${dec_model}
	fi

    if [[ -z ${device} || ${#device[@]} -eq 0 ]]; then
		if [[ ${gpu_num} -eq 0 ]]; then
			device=()
		else
        	source ./local/utils.sh
        	device=$(get_devices $gpu_num 0)
		fi
    fi
    export CUDA_VISIBLE_DEVICES=${device}

	result_file=${model_dir}/decode_result
	[[ -f ${result_file} ]] && rm ${result_file}

    test_subset=(${test_subset//,/ })
	for subset in ${test_subset[@]}; do
        subset=${subset}_st
  		cmd="python ${root_dir}/fairseq_cli/generate.py
        ${data_dir}
        --config-yaml ${data_config}
        --gen-subset ${subset}
        --task speech_to_text
        --path ${model_dir}/${dec_model}
        --results-path ${model_dir}
        --max-tokens ${max_tokens}
        --beam ${beam_size}
        --lenpen ${len_penalty}
        --scoring sacrebleu"
    	echo -e "\033[34mRun command: \n${cmd} \033[0m"

        if [[ $eval -eq 1 ]]; then
    	    eval $cmd
    	    tail -n 1 ${model_dir}/generate-${subset}.txt >> ${result_file}
        fi
	done
    cat ${result_file}
fi
