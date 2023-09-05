root=$1
working_dir=/home/domhnall/Repos/fairseq
virtual_env=/home/domhnall/Envs/fairseq/bin/activate
rel_script_path=examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py
kmeans_model_path=examples/textless_nlp/gslm/speech2unit/clustering/km.bin
acoustic_model_path=examples/textless_nlp/gslm/speech2unit/clustering/hubert_base_ls960.pt

. $virtual_env && cd $working_dir && python $rel_script_path --feature_type hubert --kmeans_model_path $kmeans_model_path --acoustic_model_path $acoustic_model_path --layer -1 --manifest_path $root/test_unit_manifest.txt --out_quantized_file_path=$root/label/test.unt --extension ".wav" --hide-fname 
