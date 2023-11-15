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
REDIS_WAIT_TIME = 0.001

TEST_VIDEO_PATH = 'datasets/example.mp4'

frontal_detector = dlib.get_frontal_face_detector()  # HOG
cnn_detector = dlib.cnn_face_detection_model_v1('avhubert/preparation/mmod_human_face_detector.dat')  # MMOD
predictor = dlib.shape_predictor('avhubert/preparation/shape_predictor_68_face_landmarks.dat')
get_detections = None

# TODO:
# server should take video path from queue instead of frames
# the point of using frames was for streaming them
# stream portions of video instead?


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


def get_video_frames(video_path):
    video_capture = cv2.VideoCapture(video_path)

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        yield frame

    video_capture.release()


def server(args):
    global get_detections
    get_detections = get_detections_cnn

    cache = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

    # do video run through
    for frame in get_video_frames(video_path=TEST_VIDEO_PATH):
        get_face_landmarks(frame=frame)

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

    cache = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

    frames = list(get_video_frames(video_path=args.video_path))

    def _profile(name, _get_detections):
        global get_detections
        get_detections = _get_detections

        # profile function call
        times = []
        for _ in range(args.num_repeats):
            start_time = time.time()

            num_fails = 0
            for frame in frames:
                rects, _ = get_face_landmarks(frame=frame)
                if not rects:
                    num_fails += 1

            times.append(time.time() - start_time)

        print(f'{name} - function call, num frames: {len(frames)}, avg. time: {np.mean(times):.2f}, fail rate: {num_fails / len(frames):.2f}')

        # profile redis call
        times = []
        for _ in range(args.num_repeats):
            start_time = time.time()

            for frame_index, frame in enumerate(frames):
                item = (frame_index, frame)
                cache.rpush(REDIS_FRAME_QUEUE, pickle.dumps(item))
            
            i, max_frames = 0, frame_index + 1
            while i < max_frames:
                item = cache.lpop(REDIS_LANDMARK_QUEUE)
                if not item:
                    continue
                _, _ = pickle.loads(item)
                i += 1  # keep popping off the list when landmarks become available

            times.append(time.time() - start_time)

        print(f'{name} - redis queue, num frames: {len(frames)}, avg. time: {np.mean(times):.2f}')

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
