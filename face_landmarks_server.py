import argparse
import os
import pickle
import time

import cv2
import dlib
import numpy as np
import redis

import config

frontal_face_detector = dlib.get_frontal_face_detector()  # HOG
cnn_face_detector = dlib.cnn_face_detection_model_v1('avhubert/preparation/mmod_human_face_detector.dat')  # MMOD
landmark_predictor = dlib.shape_predictor('avhubert/preparation/shape_predictor_68_face_landmarks.dat')
get_face_detections = None

# TODO: do we need to upscale? I don't think CFE does


def get_face_detections_frontal(frame):
    return frontal_face_detector(frame, 1)  # upsamples image 1 time to detect more faces


def get_face_detections_cnn(frame):
    rects = cnn_face_detector(frame, 1)

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
        self.pre_crop_face_scale_factor = 1.3
        self.pre_face = Rect()
        self.debug = debug

    def detect(self, frame_index, frame):
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

        if self.pre_crop_face_scale_factor > 0 and self.pre_face.top != -1:
            pre_width = self.pre_face.right - self.pre_face.left
            pre_height = self.pre_face.bottom - self.pre_face.top
            size_diff = int((pre_width * self.pre_crop_face_scale_factor) - pre_width)
            crop_x = max(0, int(self.pre_face.left - (size_diff / 2)))
            crop_y = max(0, int(self.pre_face.top - (size_diff / 2)))
            crop_width = min(processed_frame.shape[1] - crop_x, pre_width + size_diff)
            crop_height = min(processed_frame.shape[0] - crop_y, pre_height + size_diff)

            crop_rectangle = Rect(crop_x, crop_y, crop_width, crop_height)
            processed_frame = crop_rectangle.crop(frame=processed_frame)

        if self.debug:
            print(f'Frame {frame_index}: using crop_rectangle {crop_rectangle.__dict__}, processed frame {processed_frame.shape} {processed_frame.dtype}')

        faces = get_face_detections(frame=processed_frame)  # this returns list of dlib.rectangle
        if self.debug:
            print(f'Frame {frame_index}: found {len(faces)} faces')
        if len(faces) == 0:
            faces = get_face_detections(frame=frame)  # fallback to using the whole frame
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


def process_frame(frame_index, frame):
    # run face and landmark detectors
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = get_face_detections_cnn(frame=gray)
    all_coords = get_landmarks(frame=gray, faces=faces)

    # landmarks will be filtered by largest detected bbox later
    frame_landmarks = construct_landmarks_d(faces=faces, all_coords=all_coords)

    return [frame_index], [frame_landmarks]


def process_video(video_path, debug=False):
    face_detector = FaceDetector(debug=debug)
    frame_indices, video_landmarks = [], []

    for i, frame in enumerate(get_video_frames(video_path=video_path)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if i % config.FACE_DETECTION_ON_NTH_FRAME == 0:
            face = face_detector.detect(frame_index=i, frame=np.copy(gray))  # only returns 1 face

        faces = [face] if face is not None else []

        if debug:
            print(f'Frame index: {i}, using faces: {[[r.left(), r.top(), r.right(), r.bottom()] for r in faces]}')

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

        if type(item) is str:
            frame_indices, video_landmarks = process_video(video_path=item)
        else:
            frame_indices, video_landmarks = process_frame(*item)

        # queue landmarks
        for frame_index, frame_landmarks in zip(frame_indices, video_landmarks):
            item = (frame_index, frame_landmarks)
            cache.rpush(config.REDIS_LANDMARK_QUEUE, pickle.dumps(item))


# def profile(args):
#     assert os.path.exists(args.video_path)

#     cache = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT)

#     frames = list(get_video_frames(video_path=args.video_path))

#     def _profile(name, _get_detections):
#         global get_detections
#         get_detections = _get_detections

#         # profile function call
#         times = []
#         for _ in range(args.num_repeats):
#             start_time = time.time()

#             num_fails = 0
#             for frame in frames:
#                 rects, _ = get_face_landmarks(frame=frame)
#                 if not rects:
#                     num_fails += 1

#             times.append(time.time() - start_time)

#         print(f'{name} - function call, num frames: {len(frames)}, avg. time: {np.mean(times):.2f}, fail rate: {num_fails / len(frames):.2f}')

#         # profile redis call
#         times = []
#         for _ in range(args.num_repeats):
#             start_time = time.time()

#             for frame_index, frame in enumerate(frames):
#                 item = (frame_index, frame)
#                 cache.rpush(config.REDIS_FRAME_QUEUE, pickle.dumps(item))
            
#             i, max_frames = 0, frame_index + 1
#             while i < max_frames:
#                 item = cache.lpop(config.REDIS_LANDMARK_QUEUE)
#                 if not item:
#                     continue
#                 _, _ = pickle.loads(item)
#                 i += 1  # keep popping off the list when landmarks become available

#             times.append(time.time() - start_time)

#         print(f'{name} - redis queue, num frames: {len(frames)}, avg. time: {np.mean(times):.2f}')

#     _profile('DLIB frontal detector', get_face_detections_frontal)
#     _profile('DLIB MMOD CNN detector', get_face_detections_cnn)


def main(args):
    f = {
        'server': server,
        'profile': None
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
