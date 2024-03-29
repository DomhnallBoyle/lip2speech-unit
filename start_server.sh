#!/bin/bash
backend=${BACKEND:-1}
prod=${PROD:-0}
daemon=${DAEMON:-0}
decoder_cpu=${DECODER_CPU:-1}
use_email=${USE_EMAIL:-0}
fp16=${FP16:-true}

working_dir=/tmp/lip2speech
avhubert_env=${HOME}/Envs/avhubert/bin/activate
lip2speech_env=${HOME}/Envs/lip2speech-unit/bin/activate
avhubert_dir=${PWD}/avhubert/preparation
synthesis_dir=${PWD}/multi_target_lip2speech
vocoder_dir=${PWD}/multi_input_vocoder

if [[ $use_email -eq 1 ]]; then
    echo "Enter email username:"
    read email_username

    echo "Enter email password:"
    read -s email_password

    export EMAIL_USERNAME=$email_username
    export EMAIL_PASSWORD=$email_password
fi

if [[ $backend -eq 1 ]]; then

    . $avhubert_env && cd $avhubert_dir && nohup python detect_landmark_new.py --cnn_detector=mmod_human_face_detector.dat --face_predictor=shape_predictor_68_face_landmarks.dat --ffmpeg=/usr/bin/ffmpeg server >/dev/null 2>&1 &
    . $lip2speech_env && cd $synthesis_dir && nohup python inference_server.py --config-dir conf --config-name decode common.fp16=$fp16 common.user_dir=$synthesis_dir common_eval.path=$synthesis_dir/checkpoints/lip2speech_lrs3_avhubert_multi.pt common_eval.results_path=$working_dir/synthesis_results override.checkpoints_data_path=$synthesis_dir/checkpoints.json override.data=$working_dir/label override.label_dir=$working_dir/label fp16=$fp16 >/dev/null 2>&1 &
    . $lip2speech_env && cd $vocoder_dir && nohup python inference_server.py configs/lrs3/multi_input_aug.json $working_dir/synthesis_results/vocoder/label/test.tsv ../datasets/lrs3/label/dict.unt.txt --output_dir $working_dir/vocoder_results --checkpoint_file checkpoints/lrs3/multi_input/vocoder_lrs3_multi_aug.pt -n -1 >/dev/null 2>&1 &

    if [[ $decoder_cpu -eq 1 ]]; then
        . $lip2speech_env && nohup python vsg_service.py server >/dev/null 2>&1 &

        # VSG service decoder always runs on CPU
        export CUDA_VISIBLE_DEVICES=''
        . $lip2speech_env && cd $synthesis_dir && nohup python inference_server.py --config-dir conf --config-name decode common.user_dir=$synthesis_dir common_eval.path=$synthesis_dir/checkpoints/lip2speech_lrs3_avhubert_multi.pt common_eval.results_path=$working_dir/synthesis_results override.checkpoints_data_path=$synthesis_dir/checkpoints.json override.data=$working_dir/label override.label_dir=$working_dir/label port=5006 >/dev/null 2>&1 &
    fi
    
    echo -ne "Waiting for backend..."
    status_code=0
    while [[ $status_code -ne 200 ]]
    do 
        status_code=$(curl -s -o /dev/null -I -w "%{http_code}" http://127.0.0.1:5004/checkpoints)
        sleep 0.5 
    done
    echo "ready!"
fi

echo "Starting server..."

if [[ $prod -eq 1 ]]; then
    cmd="gunicorn -b 0.0.0.0:5002 --timeout 600 --keyfile key.pem --certfile cert.pem --worker-class eventlet -w 1 'server:create_app(args_path=\"server_args.json\")'"

    if [[ $daemon -eq 1 ]]; then
        cmd="${cmd} --daemon"
    fi
else
    cmd="python server.py server_args.json"
fi

eval $cmd