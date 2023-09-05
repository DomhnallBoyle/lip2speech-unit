vocoder_path=$1
output_path=$2
cuda=${CUDA_VISIBLE_DEVICES:-0}

CUDA_VISIBLE_DEVICES=$cuda \
python inference.py \
--input_code_file $vocoder_path/label/test.tsv \
--output_dir $output_path \
--checkpoint_file checkpoints/lrs3/multi_input/vocoder_lrs3_multi_aug.pt \
--config_file configs/lrs3/multi_input_aug.json \
-n -1
