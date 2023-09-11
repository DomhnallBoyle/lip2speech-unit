# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys,os,pickle,math
import cv2,dlib,skvideo,skvideo.io,time
import numpy as np
from flask import Flask, request
from http import HTTPStatus
from tqdm import tqdm


def load_video(path):
    videogen = skvideo.io.vread(path)
    frames = np.array([frame for frame in videogen])
    return frames


def detect_landmark(image, detector, cnn_detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    if len(rects) == 0:
        rects = cnn_detector(gray)
        rects = [d.rect for d in rects]
    coords = None
    for (_, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def detect_face_landmarks(face_predictor_path, cnn_detector_path, root_dir, landmark_dir, flist_fn, rank, nshard):
    detector = dlib.get_frontal_face_detector()
    cnn_detector = dlib.cnn_face_detection_model_v1(cnn_detector_path)
    predictor = dlib.shape_predictor(face_predictor_path)
    input_dir = root_dir #
    output_dir = landmark_dir #
    fids = [ln.strip() for ln in open(flist_fn).readlines()]
    num_per_shard = math.ceil(len(fids)/nshard)
    start_id, end_id = num_per_shard*rank, num_per_shard*(rank+1)
    fids = fids[start_id: end_id]
    print(f"{len(fids)} files")
    for fid in tqdm(fids):
        output_fn = os.path.join(output_dir, fid+'.pkl')
        video_path = os.path.join(input_dir, fid+'.mp4')
        frames = load_video(video_path)
        landmarks = []
        for frame in frames:
            landmark = detect_landmark(frame, detector, cnn_detector, predictor)
            landmarks.append(landmark)
        os.makedirs(os.path.dirname(output_fn), exist_ok=True)
        pickle.dump(landmarks, open(output_fn, 'wb'))
    return


def from_directory(args):
    detect_face_landmarks(args.face_predictor, args.cnn_detector, args.root, args.landmark, args.manifest, args.rank, args.nshard)


def server(args):
    from types import SimpleNamespace
    from align_mouth_new import main as align_mouth
    from count_frames_new import main as count_frames

    app = Flask(__name__)

    """
    detector = dlib.get_frontal_face_detector()
    cnn_detector = dlib.cnn_face_detection_model_v1(args.cnn_detector)
    predictor = dlib.shape_predictor(args.face_predictor)
    """

    @app.post('/extract_mouth_frames')
    def extract_mouth_frames():
        input_directory = request.json['root']
        output_directory = os.path.join(input_directory, 'landmark')
        manifest_path = os.path.join(input_directory, 'test_file.list')

        """
        with open(manifest_path, 'r') as f:
            fids = [ln.strip() for ln in f.readlines()]
       
        for fid in tqdm(fids):
            output_fn = os.path.join(output_directory, fid + '.pkl')
            video_path = os.path.join(input_directory, fid + '.mp4')
            
            frames = load_video(video_path)

            landmarks = []
            for frame in frames:
                landmark = detect_landmark(frame, detector, cnn_detector, predictor)
                landmarks.append(landmark)
       
            os.makedirs(os.path.dirname(output_fn), exist_ok=True)
            with open(output_fn, 'wb') as f:
                pickle.dump(landmarks, f)
        """

        align_mouth(SimpleNamespace(**{
            'video_direc': input_directory,
            'landmark_direc': output_directory,
            'filename_path': manifest_path,
            'save_direc': os.path.join(input_directory, 'video'),
            'mean_face': '20words_mean_face.npy',
            'crop_width': 96,
            'crop_height': 96,
            'start_idx': 48,
            'stop_idx': 68,
            'window_margin': 12,
            'ffmpeg': '/usr/bin/ffmpeg',
            'rank': 0,
            'nshard': 1
        }))

        count_frames(SimpleNamespace(**{
            'root': input_directory,
            'manifest': manifest_path,
            'rank': 0,
            'nshard': 1
        }))

        return '', HTTPStatus.NO_CONTENT

    app.run(port=args.port)


def main(args):
    skvideo.setFFmpegPath(os.path.dirname(args.ffmpeg))

    f = {
        'from_directory': from_directory,
        'server': server
    }[args.run_type](args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='detecting facial landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cnn_detector', type=str, help='path to cnn detector (download and unzip from: http://dlib.net/files/mmod_human_face_detector.dat.bz2)')
    parser.add_argument('--face_predictor', type=str, help='path to face predictor (download and unzip from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)')
    parser.add_argument('--ffmpeg', type=str, help='ffmpeg_path')

    sub_parser = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parser.add_parser('from_directory')
    parser_1.add_argument('--root', type=str, help='root dir')
    parser_1.add_argument('--landmark', type=str, help='landmark dir')
    parser_1.add_argument('--manifest', type=str, help='a list of filenames')
    parser_1.add_argument('--rank', type=int, help='rank id')
    parser_1.add_argument('--nshard', type=int, help='number of shards')
    
    parser_2 = sub_parser.add_parser('server')
    parser_2.add_argument('--port', type=int, default=5003)

    main(parser.parse_args())
