results_path=$1
label_dir=$2
working_dir=/home/domhnall/Repos/lip2speech-unit/multi_target_lip2speech
virtual_env=/home/domhnall/Envs/lip2speech-unit/bin/activate

. $virtual_env && cd $working_dir && ./scripts/lrs3/inference_avhubert.sh $results_path $label_dir
