#!/bin/bash
vocoder_path=$1
output_path=$2
checkpoint_path=$3
config_path=$4
working_dir=${PWD}/multi_input_vocoder
virtual_env=${HOME}/Envs/lip2speech-unit/bin/activate

. $virtual_env && cd $working_dir && ./scripts/lrs3/inference_aug.sh $vocoder_path $output_path $checkpoint_path $config_path
