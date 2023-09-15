label_dir=$1
output_dir=$2
batch_size=${BATCH_SIZE:-4}  # grad-accum is 8

PYTHONPATH=/home/domhnall/lip2speech-unit/fairseq \
fairseq-hydra-train \
--config-dir conf/lrs3 \
--config-name multi_target_avhubert \
hydra.run.dir=$output_dir \
common.user_dir=`pwd` \
model.w2v_path=`pwd`/checkpoints/large_vox_iter5.pt \
task.label_dir=$label_dir \
task.data=$label_dir \
dataset.batch_size=$batch_size