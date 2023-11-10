#!/bin/bash
label_dir=$1
results_path=$2
checkpoint_path=$3
working_dir=/home/domhnall/Repos/lip2speech-unit/multi_target_lip2speech
virtual_env=/home/domhnall/Envs/lip2speech-unit/bin/activate

. $virtual_env && cd $working_dir && ./scripts/lrs3/inference_avhubert.sh $label_dir $results_path $checkpoint_path
