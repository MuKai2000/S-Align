#! /bin/bash

data_dir=data_all_enfr_lcrm
#data_dir=data_all_200ende_lcrm
#test_subset=(test_p500_update)
#test_subset=(test_p0.9_p500)
test_subset=(tst-COMMON_st)
#test_subset=(tst-COMMON_asr)                                    # //need_check//
#test_subset=(tst-COMMON_mt)
#test_subset=(test)
#test_subset=(test_va)

# exp_name=ende_shrink_AT_window_scale1
# exp_name=ende_shrink_AT_sentence_scale5_nopad                  # //need_check//
# exp_name=ende_shrink_asronly_rec_autoreg
# exp_name=ende_shrink_v1_merge_AT_sentence_scale1_mixup_sen0.3_tok0.5_mt_0727
# exp_name=ende_shrink_v1_merge_baseline_mt_0724_2250
# exp_name=ende_shrink_v1_merge_AT_sentence_scale1.5_mt0.5_0728
#exp_name=ende_shrink_v1_merge_AT_sentence_scale1_mt_0725
#exp_name=ende_shrink_v1_merge_AT_sentence_0731_0.5mt_1.0at
#exp_name=ende_shrink_v1_merge_0801_two_cl
#exp_name=ende_shrink_v1_merge_0802_baseline
#exp_name=ende_shrink_v1_merge_0803_two_cl
#exp_name=ende_shrink_v1_merge_0803_at_sentence/
#exp_name=ende_shrink_v1_merge_0804_at_sentence_0.5mt_1.5at
#exp_name=ende_shrink_v1_merge_0804_two_cl_2.0
#exp_name=ende_shrink_v1_merge_0805_token_at
#exp_name=ende_shrink_v1_merge_0805_sequence_at
#exp_name=ende_shrink_v1_merge_0805_mixup_at
#exp_name=ende_shrink_v1_merge_0807_st
#exp_name=ende_shrink_v1_merge_0808_mixup_at
#exp_name=ende_shrink_v1_merge_0808_mixup_random
# exp_name=ende_shrink_v1_merge_0808_baseline_alpha2.0
#exp_name=ende_shrink_v1_merge_0809_AT_sentence_mixup_alpha2.0_mt0.5
# exp_name=ende_shrink_v1_merge_0810_AT_sentence_mixup_alpha2.0_mt0.5_scale1.5
# exp_name=ende_shrink_v1_merge_0810_AT_sentence_mixup0.1_0.5_alpha2.0_mt0.5
#exp_name=ende_shrink_v1_merge_large_0811_baseline_alpha1.5_mt0.5
# exp_name=ende_shrink_v1_merge_large_0812_doubleCL_alpha1.5_mt0.5
# exp_name=ende_shrink_v1_merge_large_0812_doubleCL_alpha2.5_mt0.5
# exp_name=ende_shrink_v1_merge_large_0812_AT_sentence_alpha1.5_mt0.5
# exp_name=ende_shrink_v1_merge_large_0812_AT_sentence_alpha1.5_mt0.5
# exp_name=ende_shrink_v1_merge_large_0813_AT_sentence_mixup0.5_alpha1.5_mt0.5
#exp_name=ende_shrink_v1_merge_large_0814_AT_sentence_mixup0.5_alpha1.5_mt0.5
#exp_name=ende_shrink_v1_merge_large_0814_AT_sentence_mixup0105_alpha1.5_mt0.5
#exp_name=ende_shrink_v1_merge_large_0814_AT_sentence_mixup0105_global_alpha1.5_mt0.5
#exp_name=ende_shrink_v1_merge_large_0815_AT_sentence_mixup0105_mt0102_id0_alpha1.5_mt0.5
#exp_name=ende_shrink_v1_merge_large_0816_baseline_cl_alpha1.5_mt0.5
# exp_name=ende_shrink_v1_merge_large_0816_AT_sentence_alpha2.5_mt0.5_nopad

# exp_name=ende_shrink_v2_merge_large_0818_baseline_alpha1.5_mt0.5
# exp_name=ende_shrink_v2_merge_large_0818_doubleCL_alpha1.5_mt0.5
#exp_name=ende_shrink_v2_merge_large_0819_AT_sentence_alpha1.5_mt0.5
#exp_name=ende_shrink_v2_merge_large_0819_doubleCL_AT_sentence_alpha1.5_mt0.5
#exp_name=ende_shrink_v2_merge_large_0819_AT_sentence_mixup0105_mt0102_alpha1.5_mt0.5
#exp_name=ende_shrink_v2_merge_large_0819_AT_sentence_mixup0105_mt0102_changeid_alpha1.5_mt0.5
#exp_name=ende_shrink_v2_merge_large_0820_doubleCL_AT_sentence_mixup0.5_alpha1.5_mt0.5
#exp_name=ende_shrink_v2_merge_large_0820_doubleCL_AT_token_alpha1.5_mt0.5

#exp_name=enfr_shrink_v2_merge_large_0821_baseline_alpha1.5_mt0.5
#exp_name=enfr_shrink_v2_merge_large_0822_noCL_AT_sentence_mixup0307_scale2.5_alpha0_mt0.5
#exp_name=enfr_shrink_v2_merge_wmt_0825_baseline_alpha1.5_mt0.5
#exp_name=enfr_shrink_v2_merge_wmt_0825_soft_noCL_AT_sentence_mixup0109_scale2.5_alpha0_mt0.5
#exp_name=enfr_shrink_v2_merge_wmt_0826_both_topCL_AT_sentence_mixup0109_scale2.5_alpha1.5_mt0.5_nograd
#exp_name=enfr_shrink_v2_merge_mustc_0827_both_topCL_AT_sentence_mixup0109_scale2.5_alpha1.5_mt0.5_nograd
#exp_name=enfr_shrink_v2_merge_mustc_0827_both_topCL_AT_sentence_mixup0307_scale2.5_alpha1.5_mt0.5_nograd
#exp_name=enfr_shrink_v2_merge_wmt_0829_both_topCL_AT_sentence_mixup0307_scale2.5_alpha1.5_mt0.5_nograd

#exp_name=enfr_shrink_v2_merge_wmt_0830_both_topCL_AT_sentence_mixup0109_scale3.5_alpha1.0_mt0.5_nograd

#exp_name=enfr_v4_merge_wmt_0906_shrink_soft_noCL_AT_sentence_mixup_changeid_scale3.5_alpha0_mt0.5
exp_name=enfr_v4_merge_wmt_0906_shrink_soft_noCL_AT_sentence_mixup_changeid_scale2_alpha0_mt0.5

n_average=5     # 5
beam_size=8     # 8         # //need_check//
len_penalty=1.0
max_tokens=15000
dec_model=checkpoint_best.pt
task=joint_triple_pretraining_merge                              # //need_check//       

gpu_num=1
device=7
export CUDA_VISIBLE_DEVICES=${device}           
cmd="./device_run.sh
    --stage 2
    --stop_stage 2
    --gpu_num ${gpu_num}
    --task ${task}
    --exp_name ${exp_name}
    --n_average ${n_average}
    --beam_size ${beam_size}
    --len_penalty ${len_penalty}
    --max_tokens ${max_tokens}
    --dec_model ${dec_model}
    "

if [[ -n ${data_dir} ]]; then
    cmd="$cmd --data_dir ${data_dir}"
fi
if [[ -n ${test_subset} ]]; then
    test_subset=`echo ${test_subset[*]} | sed 's/ /,/g'`
    cmd="$cmd --test_subset ${test_subset}"
fi

echo $cmd
eval $cmd
