label_dir=$1
output_dir=$2
avsr_checkpoint_path=$3
config_name=${CONFIG_NAME:-multi_target}
batch_size=${BATCH_SIZE:-4}  # grad accum is 8
max_updates=${MAX_UPDATES:-50000}
warmup_updates=${WARMUP_UPDATES:-10000}

PYTHONPATH=/home/domhnall/Repos/lip2speech-unit/fairseq \
fairseq-hydra-train \
--config-dir conf/lrs3 \
--config-name $config_name \
hydra.run.dir=$output_dir \
common.user_dir=`pwd` \
task.label_dir=$label_dir \
task.data=$label_dir \
+model.avsr_checkpoint_path=$avsr_checkpoint_path \
dataset.batch_size=$batch_size \
optimization.max_update=$max_updates \
lr_scheduler.warmup_updates=$warmup_updates
