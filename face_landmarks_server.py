import argparse
import pickle
import re
import subprocess
import time

import cv2
import dlib
import numpy as np
import redis

import config

VIDEO_INFO_COMMAND = 'ffprobe -i {{video_path}}'

frontal_face_detector = dlib.get_frontal_face_detector()  # HOG
cnn_face_detector = dlib.cnn_face_detection_model_v1('avhubert/preparation/mmod_human_face_detector.dat')  # MMOD
landmark_predictor = dlib.shape_predictor('avhubert/preparation/shape_predictor_68_face_landmarks.dat')
get_face_detections = None

# TODO: 
# use iBUG detector/predictor for VSG service
# fix scale functionality in FaceDetector - doesn't seem to work for videos > 500
# use multiple landmark predictors - would need to involve redis?


def get_face_detections_frontal(frame, upsample_num_times=0):
    return frontal_face_detector(frame, upsample_num_times=upsample_num_times)


def get_face_detections_cnn(frame, upsample_num_times=0):
    # NOTE: with SRAVI, camera will be close to the face - no upsampling needed
    # with VSG service, faces could be further away in video - upsample frame beforehand
    rects = cnn_face_detector(frame, upsample_num_times=upsample_num_times)

    return [d.rect for d in rects]


class Rect:

    def __init__(self, x=-1, y=-1, width=-1, height=-1):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.top = self.y
        self.left = self.x
        self.right = self.left + self.width
        self.bottom = self.top + self.height

    def crop(self, frame):
        return frame[self.top:self.bottom, self.left:self.right]


class FaceDetector:
    
    def __init__(self, debug=False):
        self.max_size = 500
        self.pre_face = Rect()
        self.debug = debug

    def detect(self, frame_index, frame, upsample_num_times=0):
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        scale = 1.0
        height, width = frame.shape
        size = max(height, width)

        if self.debug:
            print(f'Frame {frame_index}: {frame.shape}')

        if size > self.max_size:
            scale = self.max_size / size
            frame = cv2.resize(frame, (0, 0), scale, scale, cv2.INTER_AREA)

        processed_frame = np.copy(frame)
        crop_rectangle = Rect(0, 0, 0, 0)  # x, y, width, height

        if self.debug:
            print(f'Frame {frame_index}: using crop_rectangle {crop_rectangle.__dict__}, processed frame {processed_frame.shape} {processed_frame.dtype}')

        if config.FACE_DETECTION_PRE_CROP_FACE_SCALE_FACTOR > 0 and self.pre_face.top != -1:
            pre_width = self.pre_face.right - self.pre_face.left
            pre_height = self.pre_face.bottom - self.pre_face.top
            size_diff = int((pre_width * config.FACE_DETECTION_PRE_CROP_FACE_SCALE_FACTOR) - pre_width)
            crop_x = max(0, int(self.pre_face.left - (size_diff / 2)))
            crop_y = max(0, int(self.pre_face.top - (size_diff / 2)))
            crop_width = min(processed_frame.shape[1] - crop_x, pre_width + size_diff)
            crop_height = min(processed_frame.shape[0] - crop_y, pre_height + size_diff)

            crop_rectangle = Rect(crop_x, crop_y, crop_width, crop_height)
            processed_frame = crop_rectangle.crop(frame=processed_frame)

        if self.debug:
            print(f'Frame {frame_index}: using crop_rectangle {crop_rectangle.__dict__}, processed frame {processed_frame.shape} {processed_frame.dtype}')

        faces = get_face_detections(frame=processed_frame, upsample_num_times=upsample_num_times)  # this returns list of dlib.rectangle
        if self.debug:
            print(f'Frame {frame_index}: found {len(faces)} faces')
        if len(faces) == 0:
            faces = get_face_detections(frame=frame, upsample_num_times=1)  # fallback to using the whole frame and upsample
            crop_rectangle = Rect(0, 0, 0, 0)
            if self.debug:
                print(f'Frame {frame_index}: found {len(faces)} faces')

        # focus on face with largest bbox
        max_area, max_index = -1, -1
        for i, face in enumerate(faces):
            current_area = int(face.width() * face.height())
            if (current_area > max_area):
                max_area = current_area
                max_index = i
        
        if self.debug:
            print(f'Frame {frame_index}: max area {max_area} @ {max_index}')

        if max_index == -1:
            self.pre_face = Rect()
            return

        max_face = faces[max_index]

        face = dlib.rectangle(
            left=max(int((max_face.left() + crop_rectangle.x) / scale), 0),
            top=max(int((max_face.top() + crop_rectangle.y) / scale), 0),
            right=min(int((max_face.right() + crop_rectangle.x) / scale), width - 1),
            bottom=min(int((max_face.bottom() + crop_rectangle.y) / scale), height - 1)
        )

        self.pre_face.bottom = min(int(max_face.bottom() + crop_rectangle.y), int((height * scale) - 1))
        self.pre_face.top = max(int(max_face.top() + crop_rectangle.y), 0)
        self.pre_face.left = max(int(max_face.left() + crop_rectangle.x), 0)
        self.pre_face.right = min(int(max_face.right() + crop_rectangle.x), int((width * scale) - 1))

        return face


def get_landmarks(frame, faces):
    all_coords = []
    for face in faces:
        shape = landmark_predictor(frame, face)

        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y) 
        all_coords.append(coords.tolist())
    
    return all_coords


def get_video_rotation(video_path):
    cmd = VIDEO_INFO_COMMAND.format(input_video_path=video_path)

    _, stderr = subprocess.Popen(
        cmd.split(' '),
        stderr=subprocess.PIPE,
        close_fds=True
    ).communicate()

    try:
        reo_rotation = re.compile('rotate\s+:\s(\d+)')
        match_rotation = reo_rotation.search(str(stderr))
        rotation = match_rotation.groups()[0]
    except AttributeError:
        # print(f'Rotation not found: {video_path}')
        return 0

    return int(rotation)


def fix_frame_rotation(frame, rotation):
    fix_rotations_mapping = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE
    }

    new_rotation = fix_rotations_mapping.get(rotation)
    if new_rotation:
        frame = cv2.rotate(frame, new_rotation)

    return frame


def get_video_frames(video_path):
    video_capture = cv2.VideoCapture(video_path)

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        yield frame

    video_capture.release()


def construct_landmarks_d(faces, all_coords):
    return {
        'bbox': [[r.left(), r.top(), r.right(), r.bottom()] for r in faces],
        'landmarks': all_coords,
        'landmarks_scores': [[1.0] * 68 for _ in range(len(faces))]
    }


def process_frame(frame_index, frame, face_close_up=True):
    # run face and landmark detectors
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = get_face_detections_cnn(frame=gray, upsample_num_times=0 if face_close_up else 1)
    all_coords = get_landmarks(frame=gray, faces=faces)

    # landmarks will be filtered by largest detected bbox later
    frame_landmarks = construct_landmarks_d(faces=faces, all_coords=all_coords)

    return [frame_index], [frame_landmarks]


def process_video(video_path, face_close_up=True):
    face_detector = FaceDetector(debug=config.DEBUG)
    frame_indices, video_landmarks = [], []
    upsample_num_times = 0 if face_close_up else 1

    print(f'Processing video {video_path} with face close-up {face_close_up}, upsampling {upsample_num_times} times...')

    for i, frame in enumerate(get_video_frames(video_path=video_path)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if i % config.FACE_DETECTION_ON_NTH_FRAME == 0:
            face = face_detector.detect(
                frame_index=i, 
                frame=np.copy(gray),
                upsample_num_times=upsample_num_times
            )  # only returns 1 face

        faces = [face] if face is not None else []

        if config.DEBUG:
            print(f'Frame index: {i}, {gray.shape}, using faces: {[[r.left(), r.top(), r.right(), r.bottom()] for r in faces]}')

        all_coords = get_landmarks(frame=gray, faces=faces)

        frame_landmarks = construct_landmarks_d(faces=faces, all_coords=all_coords)

        frame_indices.append(i)
        video_landmarks.append(frame_landmarks)

    return frame_indices, video_landmarks


def server(args):
    global get_face_detections
    get_face_detections = get_face_detections_cnn

    cache = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT)

    # do video run through to prepare GPU
    for i, frame in enumerate(get_video_frames(video_path=config.TEST_VIDEO_PATH)):
        process_frame(frame_index=i, frame=frame)
    
    while True: 
        item = cache.lpop(config.REDIS_VIDEO_QUEUE) or cache.lpop(config.REDIS_FRAME_QUEUE)
        if not item:
            time.sleep(config.REDIS_LANDMARK_WAIT_TIME)
            continue

        item = pickle.loads(item)

        # process media differently based on no. of args
        if len(item) == 2:
            frame_indices, video_landmarks = process_video(*item)
        else:
            frame_indices, video_landmarks = process_frame(*item)

        # queue landmarks
        for frame_index, frame_landmarks in zip(frame_indices, video_landmarks):
            item = (frame_index, frame_landmarks)
            cache.rpush(config.REDIS_LANDMARK_QUEUE, pickle.dumps(item))


def profile(args):
    global get_face_detections
    get_face_detections = get_face_detections_cnn
    face_detector = FaceDetector()

    total_times = []
    video_face_times, video_landmark_times = [], []
    for _ in range(args.num_repeats):
        round_start_time = time.time()

        frame_face_times, frame_landmark_times = [], []
        for i, frame in enumerate(get_video_frames(video_path=args.video_path)):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if i % args.skip_nth_frame == 0:
                start_time = time.time()
                face = face_detector.detect(
                    frame_index=i, 
                    frame=np.copy(gray), 
                    upsample_num_times=args.upsample_num_times
                )  # only returns 1 face
                frame_face_times.append(time.time() - start_time)

            faces = [face] if face is not None else []
            
            start_time = time.time()
            get_landmarks(frame=gray, faces=faces)
            frame_landmark_times.append(time.time() - start_time)

        total_times.append(time.time() - round_start_time)
        video_face_times.append(sum(frame_face_times))
        video_landmark_times.append(sum(frame_landmark_times))

    print(f'Face Detection: {np.mean(video_face_times):.2f}')
    print(f'Landmark Detection: {np.mean(video_landmark_times):.2f}')
    print(f'Total: {np.mean(total_times):.2f}')


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
    parser_2.add_argument('--skip_nth_frame', type=int, default=1)
    parser_2.add_argument('--upsample_num_times', type=int, default=0)
    parser_2.add_argument('--num_repeats', type=int, default=50)

    main(parser.parse_args())
