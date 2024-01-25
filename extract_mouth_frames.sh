#!/bin/bash
root=$1
type=${TYPE:-test}
detect_landmark=${DETECT_LANDMARK:-0}
align_mouth=${ALIGN_MOUTH:-0}
working_dir=${PWD}/avhubert/preparation
virtual_env=${HOME}/Envs/avhubert/bin/activate

if [[ $detect_landmark -eq 1 ]]; then
    . $virtual_env && cd $working_dir && python detect_landmark.py --root $root --cnn_detector mmod_human_face_detector.dat --face_predictor shape_predictor_68_face_landmarks.dat --ffmpeg /usr/bin/ffmpeg --manifest $root/${type}_file.list --nshard 1 --rank 0 --landmark $root/landmark
fi

if [[ $align_mouth -eq 1 ]]; then
    . $virtual_env && cd $working_dir && python align_mouth.py --video-direc $root --landmark-direc $root/landmark --filename-path $root/${type}_file.list --save-direc $root/video --mean-face 20words_mean_face.npy --ffmpeg /usr/bin/ffmpeg --rank 0 --nshard 1
fi

. $virtual_env && cd $working_dir && python count_frames.py --root $root --manifest $root/${type}_file.list --nshard 1 --rank 0
