vocoder_path=$1
output_path=$2
working_dir=/home/domhnall/Repos/lip2speech-unit/multi_input_vocoder
virtual_env=/home/domhnall/Envs/lip2speech-unit/bin/activate

. $virtual_env && cd $working_dir && ./scripts/lrs3/inference_aug.sh $vocoder_path $output_path
