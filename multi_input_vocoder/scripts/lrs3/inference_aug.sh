vocoder_path=$1
output_path=$2
checkpoint_path=${3:-`pwd`/checkpoints/lrs3/multi_input/vocoder_lrs3_multi_aug.pt}
config_path=${4:-`pwd`/configs/lrs3/multi_input_aug.json}
cuda=${CUDA_VISIBLE_DEVICES:-0}

CUDA_VISIBLE_DEVICES=$cuda \
python inference.py \
$config_path \
$vocoder_path/label/test.tsv \
../datasets/lrs3/label/dict.unt.txt \
--output_dir $output_path \
--checkpoint_file $checkpoint_path \
-n -1
