import argparse
import os
import pickle
import time

import cv2
import dlib
import numpy as np
import redis

REDIS_HOST = os.environ.get('REDIS_HOST', '127.0.0.1')
REDIS_PORT = 6379
REDIS_FRAME_QUEUE = 'vsg_frame_queue'
REDIS_LANDMARK_QUEUE = 'vsg_landmark_queue'
REDIS_WAIT_TIME = 0.01

frontal_detector = dlib.get_frontal_face_detector()  # HOG
cnn_detector = dlib.cnn_face_detection_model_v1('avhubert/preparation/mmod_human_face_detector.dat')  # MMOD
predictor = dlib.shape_predictor('avhubert/preparation/shape_predictor_68_face_landmarks.dat')
get_detections = None

# TODO: 
# fallback to ibug detector if missing detections?
# will HOG be faster than MMOD on production server? Should use MMOD as backup if so
# 1) HOG + ibug fallback?
# 2) HOG + MMOD fallback?


def get_detections_frontal(frame):
    return frontal_detector(frame, 1)  # upsamples image 1 time to detect more faces


def get_detections_cnn(frame):
    rects = cnn_detector(frame, 1)

    return [d.rect for d in rects]


def get_face_landmarks(frame):
    # convert to grayscale and detect multiple faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = get_detections(frame=gray)

    all_coords = []
    for rect in rects:
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y) 
        all_coords.append(coords.tolist())

    return rects, all_coords


def server(args):
    global get_detections
    get_detections = get_detections_cnn

    cache = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

    while True: 
        item = cache.lpop(REDIS_FRAME_QUEUE)
        if not item:
            time.sleep(REDIS_WAIT_TIME)
            continue

        frame_index, frame = pickle.loads(item)

        # run face and landmark detectors
        rects, all_coords = get_face_landmarks(frame=frame)
        
        # landmarks will be filtered by largest detected bbox later
        frame_landmarks = {
            'bbox': [[r.left(), r.top(), r.right(), r.bottom()] for r in rects],
            'landmarks': all_coords,
            'landmarks_scores': [[1.0] * 68 for _ in range(len(rects))]
        }

        item = (frame_index, frame_landmarks)
        cache.rpush(REDIS_LANDMARK_QUEUE, pickle.dumps(item))


def profile(args):
    assert os.path.exists(args.video_path)

    def get_video_frames(video_path):
        video_capture = cv2.VideoCapture(video_path)

        while True:
            success, frame = video_capture.read()
            if not success:
                break

            yield frame

        video_capture.release()

    frames = list(get_video_frames(video_path=args.video_path))

    def _profile(name, _get_detections):
        global get_detections
        get_detections = _get_detections

        times = []
        for _ in range(args.num_repeats):
            start_time = time.time()

            num_fails = 0
            for frame in frames:
                rects, _ = get_face_landmarks(frame=frame)
                if not rects:
                    num_fails += 1

            times.append(time.time() - start_time)
        print(f'{name}, avg. time: {np.mean(times):.2f}, fail rate: {num_fails / len(frames):.2f}')

    _profile('DLIB frontal detector', get_detections_frontal)
    _profile('DLIB MMOD CNN detector', get_detections_cnn)


def main(args):
    f = {
        'server': server,
        'profile': profile
    }
    f[args.run_type](args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('server')

    parser_2 = sub_parsers.add_parser('profile')
    parser_2.add_argument('video_path')
    parser_2.add_argument('--num_repeats', type=int, default=50)

    main(parser.parse_args())
