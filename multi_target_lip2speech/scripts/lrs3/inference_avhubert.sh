label_dir=$1
results_path=$2
checkpoint_path=${3:-`pwd`/checkpoints/lip2speech_lrs3_avhubert_multi.pt}
cuda=${CUDA_VISIBLE_DEVICES:-0}

CUDA_VISIBLE_DEVICES=$cuda \
PYTHONPATH=${HOME}/Repos/lip2speech-unit/fairseq \
python -B inference.py \
--config-dir conf \
--config-name decode \
common.user_dir=`pwd` \
common_eval.path=$checkpoint_path \
common_eval.results_path=$results_path \
override.data=$label_dir \
override.label_dir=$label_dir
