import argparse
import json
import logging
import shutil
import sys
import time
import traceback
import uuid
from http import HTTPStatus
from pathlib import Path

import redis
import requests

import config
from email_client import send_email
from helpers import WhisperASR, crop_video, extract_audio, get_video_duration, merge_videos

MAX_VIDEO_DURATION = config.MAX_VIDEO_DURATION - 0.5  # bit of leeway for decoder success
MIN_VIDEO_DURATION = 1
SLEEP_TIME = 10

asr = WhisperASR(model='medium.en', device='cpu')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler('vsg_service.log'),
        logging.StreamHandler(sys.stdout)
    ]
)


def synthesise(url, video_path, audio_path):
    with open(video_path, 'rb') as f:
        files = {'video': f.read()}

    if audio_path:
        with open(audio_path, 'rb') as f:
            files['audio'] = f.read()

    return requests.post(f'{url}/synthesise?asr=0', files=files, verify=False)


def service(url, uid, video_path, audio_path, notify=False, **kwargs):
    # setup
    output_directory = Path('/tmp/vsg')
    if output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir()

    shutil.copyfile(video_path, output_directory.joinpath('input.mp4'))

    file_list_path = output_directory.joinpath('file_list.txt')

    video_duration = get_video_duration(video_path)

    start_time = time.time()

    crop_start_time, cropped_video_index = 0, 0
    while crop_start_time < video_duration:
        crop_end_time = crop_start_time + MAX_VIDEO_DURATION
        if crop_end_time > video_duration:
            crop_end_time = video_duration

        crop_duration = crop_end_time - crop_start_time
        assert crop_duration <= MAX_VIDEO_DURATION

        # e.g. sometimes you can get corrupted crop if start and end time is 15.5 and 15.503
        if crop_duration < MIN_VIDEO_DURATION:
            break

        logging.info(f'{uid} - synthesising segment {cropped_video_index + 1} from {crop_start_time} - {crop_end_time}...')

        # crop video to start and end time
        cropped_video_path = f'/tmp/cropped_video_{cropped_video_index}.mp4'
        crop_video(video_path, crop_start_time, crop_end_time, cropped_video_path)

        # synthesise cropped video
        response = synthesise(url=url, video_path=cropped_video_path, audio_path=audio_path)
        if response.status_code != HTTPStatus.OK:
            logging.error(f'{uid} - Failed synthesis: {response.text}')
            break
        video_id = response.json()['videoId']

        # save results
        synthesised_video_path = f'static/{video_id}.mp4'
        output_segment_path = output_directory.joinpath(f'{cropped_video_index}.mp4')
        shutil.copyfile(synthesised_video_path, output_segment_path)

        with file_list_path.open('a') as f:
            f.write(f'file {output_segment_path.name}\n')

        crop_start_time = crop_end_time
        cropped_video_index += 1

        time.sleep(SLEEP_TIME)  # give other users a chance

    if response.status_code != HTTPStatus.OK:
        if notify:
            send_email(
                sender=config.EMAIL_USERNAME,
                receivers=config.EMAIL_RECEIVERS,
                subject=f'VSG Service Response - {uid}',
                content=f'The video has failed, please check the logs.'
            )
        return

    # stitch synthesised videos together
    logging.info(f'{uid} - stitching segments...')
    stitched_output_path = f'static/{uid}.mp4'
    merge_videos(file_list_path=file_list_path, output_video_path=stitched_output_path)

    # run Whisper ASR on the full stitched audio
    logging.info(f'{uid} - running Whisper ASR...')
    stitched_audio_path = str(output_directory.joinpath('audio.wav'))
    extract_audio(video_path=stitched_output_path, audio_path=stitched_audio_path)
    asr_transcriptions = asr.run(stitched_audio_path)
    with open(f'static/{uid}.txt', 'w') as f:
        for t in asr_transcriptions:
            f.write(f'{t}\n')

    time_taken = time.time() - start_time
    time_taken = round(time_taken / 60, 2)

    logging.info(f'{uid} - done, took {time_taken} mins')

    # notify liopa personnel of completion
    if notify:
        send_email(
            sender=config.EMAIL_USERNAME,
            receivers=config.EMAIL_RECEIVERS,
            subject=f'VSG Service Response - {uid}',
            content=f'The video has been synthesised successfully.'
        )


def server(**kwargs):
    cache = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT)

    # this should pick off any items in the queue if they haven't been processed
    while True:
        service_request = cache.lpop(config.REDIS_SERVICE_QUEUE)
        if not service_request:
            time.sleep(config.REDIS_VSG_SERVICE_WAIT_TIME)
            continue

        try:
            service(**json.loads(service_request), notify=True)
        except Exception:
            logging.error(traceback.format_exc())


def email_client(recipient, url, video_id, **kwargs):
    video_url = f'{url}/static/{video_id}.mp4'
    asr_transcriptions_url = f'{url}/static/{video_id}.txt'

    send_email(
        sender=config.EMAIL_USERNAME,
        receivers=[recipient],
        subject=f'VSG Service Response - {video_id}',
        content=f'The video has been synthesised successfully.\nThe video can be downloaded from {video_url}\nThe ASR transcriptions can be downloaded from {asr_transcriptions_url}'
    )


def main(args):
    f = {
        'service': service,
        'server': server,
        'email_client': email_client
    }
    f[args.run_type](**args.__dict__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('server')

    parser_2 = sub_parsers.add_parser('service')
    parser_2.add_argument('url')
    parser_2.add_argument('video_path')
    parser_2.add_argument('--uid', default=str(uuid.uuid4()))
    parser_2.add_argument('--audio_path')

    parser_3 = sub_parsers.add_parser('email_client')
    parser_3.add_argument('recipient')
    parser_3.add_argument('url')
    parser_3.add_argument('video_id')

    main(parser.parse_args())
