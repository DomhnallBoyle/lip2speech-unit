root=$1
working_dir=/home/domhnall/Repos/lip2speech-unit/avhubert/preparation
virtual_env=/home/domhnall/Envs/avhubert/bin/activate

# . $virtual_env && cd $working_dir && python detect_landmark.py --root $root --cnn_detector mmod_human_face_detector.dat --face_predictor shape_predictor_68_face_landmarks.dat --ffmpeg /usr/bin/ffmpeg --manifest $root/test_file.list --nshard 1 --rank 0 --landmark $root/landmark
. $virtual_env && cd $working_dir && python align_mouth.py --video-direc $root --landmark-direc $root/landmark --filename-path $root/test_file.list --save-direc $root/video --mean-face 20words_mean_face.npy --ffmpeg /usr/bin/ffmpeg --rank 0 --nshard 1
. $virtual_env && cd $working_dir && python count_frames.py --root $root --manifest $root/test_file.list --nshard 1 --rank 0
