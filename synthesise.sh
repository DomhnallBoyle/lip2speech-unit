#!/bin/bash
label_dir=$1
results_path=$2
checkpoint_path=$3
extra_state=$4
working_dir=${PWD}/multi_target_lip2speech
virtual_env=${HOME}/Envs/lip2speech-unit/bin/activate

# . $virtual_env && cd $working_dir && ./scripts/lrs3/inference_avhubert.sh $label_dir $results_path $checkpoint_path
 . $virtual_env && cd $working_dir && ./scripts/lrs3/inference.sh $label_dir $results_path $checkpoint_path $extra_state
