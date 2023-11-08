import argparse
from http import HTTPStatus

import requests

import config
from helpers import get_video_frames, save_video


def main(args):
    frame = next(get_video_frames(video_path=args.video_path))

    video_path = '/tmp/video.mp4'
    duration = 1

    while True:
        print(f'Attempting {duration} seconds...')

        # increase video duration by 1 second each time
        frames = [frame] * config.FPS * duration
        save_video(frames, video_path, config.FPS, colour=True)

        # synthesise - not running ASR, ASR GPU memory shouldn't increase anyway
        # video is resized the same so detector GPU memory shouldn't increase either
        with open(video_path, 'rb') as f:
            response = requests.post(f'{args.url}/synthesise?asr=0', files={'video': f.read()}, verify=False)
        
        if response.status_code != HTTPStatus.OK:
            print('Failed:', response.content)
            break

        duration += 1

    print(f'Duration limit on GPU: {duration - 1} seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('url')
    parser.add_argument('video_path')

    main(parser.parse_args())
