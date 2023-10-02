backend=${BACKEND:-1}
prod=${PROD:-0}
daemon=${DAEMON:-0}
working_dir=/tmp/lip2speech
avhubert_env=/home/domhnall/Envs/avhubert/bin/activate
lip2speech_env=/home/domhnall/Envs/lip2speech-unit/bin/activate
avhubert_dir=/home/domhnall/Repos/lip2speech-unit/avhubert/preparation
synthesis_dir=/home/domhnall/Repos/lip2speech-unit/multi_target_lip2speech
vocoder_dir=/home/domhnall/Repos/lip2speech-unit/multi_input_vocoder

if [[ $backend -eq 1 ]]; then
    . $avhubert_env && cd $avhubert_dir && nohup python detect_landmark_new.py --cnn_detector=mmod_human_face_detector.dat --face_predictor=shape_predictor_68_face_landmarks.dat --ffmpeg=/usr/bin/ffmpeg server >/dev/null 2>&1 &
    . $lip2speech_env && cd $synthesis_dir && nohup python inference_new.py --config-dir conf --config-name decode common.user_dir=$synthesis_dir common_eval.path=$synthesis_dir/checkpoints/lip2speech_lrs3_avhubert_multi.pt common_eval.results_path=$working_dir/synthesis_results override.checkpoints_data_path=$synthesis_dir/checkpoints.json override.data=$working_dir/label override.label_dir=$working_dir/label >/dev/null 2>&1 &
    . $lip2speech_env && cd $vocoder_dir && nohup python inference_new.py --input_code_file $working_dir/synthesis_results/vocoder/label/test.tsv --output_dir $working_dir/vocoder_results --checkpoint_file checkpoints/lrs3/multi_input/vocoder_lrs3_multi_aug.pt --config_file configs/lrs3/multi_input_aug.json -n -1 >/dev/null 2>&1 &
    
    echo -ne "Waiting for backend..."
    status_code=0;
    while [[ $status_code -ne 200 ]]
    do 
        status_code=$(curl -s -o /dev/null -I -w "%{http_code}" http://127.0.0.1:5004/checkpoints);
        sleep 2;
    done
    echo "ready!";
fi

echo "Starting server..."

if [[ $prod -eq 1 ]]; then
    cmd="gunicorn -b 0.0.0.0:5002 --timeout 600 --keyfile key.pem --certfile cert.pem 'server:web_app(args_path=\"server_args.json\")'"

    if [[ $daemon -eq 1 ]]; then
        cmd="${cmd} --daemon"
    fi
else
    cmd="python server.py $1 --debug"
fi

eval $cmd