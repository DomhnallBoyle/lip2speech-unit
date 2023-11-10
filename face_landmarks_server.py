import pickle
import time

import redis

import config
from helpers import get_face_landmarks, init_facial_detectors


def main():
    # pre-run the model loading for extracting landmarks which takes a while on first run
    init_facial_detectors(video_path='datasets/example.mp4')

    cache = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT)

    while True: 
        item = cache.lpop(config.REDIS_FRAME_QUEUE)
        if not item:
            time.sleep(config.REDIS_WAIT_TIME)
            continue

        frame_index, frame = pickle.loads(item)

        _, frame_landmarks = get_face_landmarks(video_frames=[frame], _filter=False)

        for l in ['bbox', 'landmarks', 'landmarks_scores']:
            frame_landmarks[l] = [frame_landmarks[l][0].tolist()]

        item = (frame_index, frame_landmarks)
        cache.rpush(config.REDIS_LANDMARK_QUEUE, pickle.dumps(item))


if __name__ == '__main__':
    main()
