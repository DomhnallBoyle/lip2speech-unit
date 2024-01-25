#!/bin/bash
root=$1
type=${TYPE:-test}
working_dir=${HOME}/Repos/fairseq
virtual_env=${HOME}/Envs/fairseq/bin/activate
rel_script_path=examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py
kmeans_model_path=examples/textless_nlp/gslm/speech2unit/clustering/km.bin
acoustic_model_path=examples/textless_nlp/gslm/speech2unit/clustering/hubert_base_ls960.pt

. $virtual_env && cd $working_dir && python $rel_script_path --feature_type hubert --kmeans_model_path $kmeans_model_path --acoustic_model_path $acoustic_model_path --layer -1 --manifest_path $root/${type}_unit_manifest.txt --out_quantized_file_path=$root/label/${type}.unt --extension ".wav" --hide-fname 
