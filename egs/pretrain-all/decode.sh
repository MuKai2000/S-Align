#! /bin/bash

data_dir=data_all_ende_lcrm
#test_subset=(tst-COMMON_st)
test_subset=(tst-COMMON_asr)                  
#test_subset=(tst-COMMON_mt)

exp_name=

n_average=5     
beam_size=8         
len_penalty=1.0
max_tokens=15000
dec_model=checkpoint_best.pt
task=joint_triple_pretraining_merge                             

gpu_num=1
device=2
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
