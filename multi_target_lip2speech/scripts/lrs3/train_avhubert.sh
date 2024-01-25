label_dir=$1
output_dir=$2
checkpoint_path=$3
batch_size=${BATCH_SIZE:-4}  # grad-accum is 8
max_updates=${MAX_UPDATES:-50000}
warmup_updates=${WARMUP_UPDATES:-10000}

# freeze_finetune_updates == max_updates ensures AV-Hubert pretrained visual frontend is frozen

PYTHONPATH=${HOME}/Repos/lip2speech-unit/fairseq \
fairseq-hydra-train \
--config-dir conf/lrs3 \
--config-name multi_target_avhubert \
hydra.run.dir=$output_dir \
common.user_dir=`pwd` \
model.w2v_path=`pwd`/checkpoints/large_vox_iter5.pt \
model.checkpoint_path=$checkpoint_path \
task.label_dir=$label_dir \
task.data=$label_dir \
dataset.batch_size=$batch_size \
optimization.max_update=$max_updates \
model.freeze_finetune_updates=$max_updates \
lr_scheduler.warmup_updates=$warmup_updates