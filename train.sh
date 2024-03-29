label_dir=$1
output_dir=$2
extra_state=$3
working_dir=${PWD}/multi_target_lip2speech
virtual_env=${HOME}/Envs/lip2speech-unit/bin/activate
config_name=${CONFIG_NAME:-multi_target}
batch_size=${BATCH_SIZE:-4}  # grad accum is 8
max_updates=${MAX_UPDATES:-50000}
warmup_updates=${WARMUP_UPDATES:-10000}

. $virtual_env && cd $working_dir && CONFIG_NAME=$config_name BATCH_SIZE=$batch_size MAX_UPDATES=$max_updates WARMUP_UPDATES=$warmup_updates ./scripts/lrs3/train.sh $label_dir $output_dir $extra_state
