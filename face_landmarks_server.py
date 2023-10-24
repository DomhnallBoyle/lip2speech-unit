import json
import sys
import time

import cv2
import numpy as np
import redis

import config
sys.path.append('/home/domhnall/Repos/sv2s')
from detector import get_face_landmarks

WAIT_TIME = 0.01


def bytes_to_frame(encoded_frame):
    return cv2.imdecode(np.frombuffer(encoded_frame, dtype=np.uint8), cv2.IMREAD_COLOR)


def main():
    # pre-run the model loading for extracting landmarks which takes a while on first run
    get_face_landmarks(video_path='datasets/example.mp4')

    cache = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT)
    
    for q in [config.REDIS_FRAME_QUEUE, config.REDIS_LANDMARK_QUEUE]:
        cache.delete(q)

    while True: 
        encoded_frame = cache.lpop(config.REDIS_FRAME_QUEUE)
        if not encoded_frame:
            print('Waiting...')
            time.sleep(WAIT_TIME)
            continue
    
        frame = bytes_to_frame(encoded_frame=encoded_frame)
        _, frame_landmarks = get_face_landmarks(video_frames=[frame], _filter=False)

        for l in ['bbox', 'landmarks', 'landmarks_scores']:
            frame_landmarks[l] = [frame_landmarks[l][0].tolist()]

        cache.rpush(config.REDIS_LANDMARK_QUEUE, json.dumps(frame_landmarks))


if __name__ == '__main__':
    main()
