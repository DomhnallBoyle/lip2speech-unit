import argparse
import json
import logging
import pickle
import shutil
import sqlite3
import subprocess
import sys
import time
import threading
import traceback
import uuid
from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import requests
import soundfile as sf
import torch
from flask import Flask, Response, g as app_context, render_template, request, url_for
from werkzeug.datastructures import FileStorage

from db import DB
from create_dataset import manifests as create_manifests, vocoder as setup_vocoder_inference
sys.path.append('/home/domhnall/Repos/sv2s')
from asr import WhisperASR
from audio_utils import preprocess_audio as post_process_audio
from detector import get_face_landmarks
from utils import convert_fps, convert_video_codecs, get_fps, get_speaker_embedding, get_video_frames, overlay_audio

FPS = 25
SAMPLING_RATE = 16000
AUDIO_FRAME_RATE = 50
MAX_HEIGHT, MAX_WIDTH = 480, 640
MAX_GPU_DURATION = 6
USING_GPU = torch.cuda.is_available()
DB_PATH = Path('server.db')
STATIC_PATH = Path('static')
SERVER_PATH = Path('/tmp/server')
INPUTS_PATH = SERVER_PATH.joinpath('inputs')
WORKING_DIRECTORY = Path('/tmp/lip2speech')
POST_PROCESSED_AUDIO_PATH = Path('/tmp/post_processed_audio.wav')
TYPE = 'test'
VIDEO_RAW_DIRECTORY = WORKING_DIRECTORY.joinpath(f'{TYPE}')
AUDIO_DIRECTORY = WORKING_DIRECTORY.joinpath(f'audio/{TYPE}')
VIDEO_DIRECTORY = WORKING_DIRECTORY.joinpath(f'video/{TYPE}')
MEL_SPEC_DIRECTORY = WORKING_DIRECTORY.joinpath(f'mel/{TYPE}')
SPK_EMB_DIRECTORY = WORKING_DIRECTORY.joinpath(f'spk_emb/{TYPE}')
LANDMARKS_DIRECTORY = WORKING_DIRECTORY.joinpath(f'landmark/{TYPE}')
LABEL_DIRECTORY = WORKING_DIRECTORY.joinpath('label')
SYNTHESIS_DIRECTORY = WORKING_DIRECTORY.joinpath('synthesis_results')
VOCODER_DIRECTORY = WORKING_DIRECTORY.joinpath('vocoder_results')

device = torch.device('cuda:0' if USING_GPU else 'cpu')
asr = WhisperASR(model='small', device=device)
sem = threading.Semaphore()
speaker_embedding_lookup = {}
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.info(f'Using: {device}')


def log_except_hook(*exc_info):
    text = ''.join(traceback.format_exception(*exc_info))
    logging.error(f'Unhandled exception: {text}') 


sys.excepthook = log_except_hook


def run_command(s):
    subprocess.run(s, shell=True)


def resize_video(video_path, width, height, output_video_path='/tmp/video_resized.mp4'):
    run_command(f'ffmpeg -y -i {video_path} -vf scale="{width}:{height}" {output_video_path} -loglevel quiet')
    shutil.copyfile(output_video_path, video_path)


def get_video_size(video_path):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    return width, height


def execute_request(name, r, *args, **kwargs):
    start_time = time.time()
    try:
        response = r(*args, **kwargs)
    except Exception as e:
        logging.error(f'Request failed: {e}')
        response = Response(status=HTTPStatus.INTERNAL_SERVER_ERROR)
    time_taken = round(time.time() - start_time, 2)

    logging.info(f'Request [{name}] took {time_taken} secs')
    
    return response


def _get_speaker_embedding(audio_path):
    start_time = time.time()
    speaker_embedding = get_speaker_embedding(audio_path=audio_path)
    time_taken = round(time.time() - start_time, 2)
    logging.info(f'Extracting speaker embedding took {time_taken} secs')

    speaker_embedding = np.asarray(speaker_embedding, dtype=np.float32)
    assert speaker_embedding.shape == (256,) and speaker_embedding.dtype == np.float32

    return speaker_embedding


def web_app(args=None, args_path=None):
    if not args and not args_path:
        logging.info('Args required')
        return

    if args_path:
        with open(args_path, 'r') as f:
            args = SimpleNamespace(**json.load(f))

    assert 'audio_paths' in args.__dict__
    args.audio_paths = [p for p in args.audio_paths if p]
    assert len(args.audio_paths) > 0, f'Audio paths required'
    assert len(args.audio_paths) == len(set(args.audio_paths)), f'Non-unique audio paths'
    assert 'web_client_run_asr' in args.__dict__

    # setup static and server directories
    for d in [INPUTS_PATH, STATIC_PATH]:
        d.mkdir(parents=True, exist_ok=True)

    if not DB_PATH.exists():
        logging.info('Database does not exist, run migrations...')
        return

    # move any audio files to static and pre-load speaker embeddings for them
    for i, audio_path in enumerate(args.audio_paths):
        audio_path = Path(audio_path)
        assert audio_path.exists(), f'{audio_path} does not exist'
        new_audio_path = STATIC_PATH.joinpath(f'speaker_audio_{i + 1}.wav')
        shutil.copyfile(audio_path, new_audio_path)
        args.audio_paths[i] = str(new_audio_path)
        try:
            speaker_embedding_lookup[new_audio_path.name] = _get_speaker_embedding(audio_path=str(new_audio_path))
        except requests.exceptions.ConnectionError:
            logging.warning('Speaker embedding server is down...')
    
    # get synthesiser checkpoints
    response = execute_request('GET CHECKPOINTS', requests.get, 'http://127.0.0.1:5004/checkpoints')
    if response.status_code != HTTPStatus.OK:
        logging.warning('Synthesiser is down...')
        checkpoint_ids = []
    else:
        checkpoint_ids = response.json()

    # record the models and speaker audios
    with DB(DB_PATH) as cur:
        for checkpoint_id in checkpoint_ids:
            try:
                cur.execute('INSERT INTO model (id, name) VALUES (?, ?)', (str(uuid.uuid4()), checkpoint_id))
            except sqlite3.IntegrityError:
                pass

        for audio_path in args.audio_paths:
            try:
                cur.execute('INSERT INTO audio (id, name) VALUES (?, ?)', (str(uuid.uuid4()), Path(audio_path).name))
            except sqlite3.IntegrityError:
                pass

    app = Flask(__name__, static_folder=str(STATIC_PATH))
    app.secret_key = str(uuid.uuid4())

    @app.before_request
    def incoming_request():
        # acquire lock if applicable
        if request.endpoint == 'synthesise':
            sem.acquire()

        app_context.start_time = time.time()

    @app.after_request
    def outgoing_response(response):
        time_taken = round(time.time() - app_context.start_time, 2)
        logging.info(f'{request.method} {request.path} {dict(request.args)} {response.status_code} - {time_taken} secs')

        # release lock if applicable
        if request.endpoint == 'synthesise':
            sem.release()

        return response

    @app.get('/demo')
    def demo():
        return render_template('index.html', **{
            'audio_paths': args.audio_paths,
            'checkpoint_ids': checkpoint_ids, 
            'web_client_run_asr': int(args.web_client_run_asr)
        })

    @app.post('/synthesise')
    def synthesise():
        name = str(uuid.uuid4())
        video_upload_path = str(INPUTS_PATH.joinpath(f'{name}.mp4'))
        audio_upload_path = str(INPUTS_PATH.joinpath(f'{name}.wav'))
        video_raw_path = str(VIDEO_RAW_DIRECTORY.joinpath(f'{name}.mp4'))
        video_landmarks_path = str(LANDMARKS_DIRECTORY.joinpath(f'{name}.pkl'))
        audio_path = str(AUDIO_DIRECTORY.joinpath(f'{name}.wav'))
        pred_audio_path = str(VOCODER_DIRECTORY.joinpath(f'pred_wav/{TYPE}/{name}.wav'))
        video_download_path = str(STATIC_PATH.joinpath(f'{name}.mp4'))

        video_file = request.files['video']
        video_file.save(video_upload_path)

        # optional model checkpoint id
        checkpoint_id = request.form.get('checkpoint_id', 'base')
        response = execute_request('LOAD CHECKPOINT', requests.post, 'http://127.0.0.1:5004/load_checkpoint', json={'checkpoint_id': checkpoint_id})
        if response.status_code != HTTPStatus.NO_CONTENT:
            return {'message': 'Failed to load checkpoint'}, HTTPStatus.INTERNAL_SERVER_ERROR

        # optional audio file for speaker embedding
        audio_file = FileStorage(filename='speaker_audio_1.wav')  # default speaker embedding
        if 'audio' in request.files:
            audio_file = request.files['audio']
            audio_file.save(audio_upload_path)

        # setup directory
        if WORKING_DIRECTORY.exists():
            shutil.rmtree(WORKING_DIRECTORY)
        for d in [WORKING_DIRECTORY, VIDEO_RAW_DIRECTORY, AUDIO_DIRECTORY, VIDEO_DIRECTORY, MEL_SPEC_DIRECTORY, SPK_EMB_DIRECTORY, LANDMARKS_DIRECTORY, LABEL_DIRECTORY]:
            d.mkdir(parents=True)

        # convert size if applicable
        if USING_GPU:
            width, height = get_video_size(video_path=video_upload_path)
            if width > MAX_WIDTH or height > MAX_HEIGHT:
                resize_video(video_path=video_upload_path, width=MAX_WIDTH, height=MAX_HEIGHT)

        # convert fps if applicable
        if get_fps(video_path=video_upload_path) != FPS:
            convert_fps(input_video_path=video_upload_path, fps=FPS, output_video_path=video_raw_path)
        else:
            shutil.copyfile(video_upload_path, video_raw_path)

        num_video_frames = len(get_video_frames(video_path=video_raw_path))
        video_duration = num_video_frames / FPS

        # check video duration
        if USING_GPU and video_duration > MAX_GPU_DURATION:
            return {'message': f'Video too long, must be <= {MAX_GPU_DURATION} seconds'}, HTTPStatus.BAD_REQUEST 

        # extract audio
        num_audio_frames = int(video_duration * SAMPLING_RATE)
        audio = np.random.rand(num_audio_frames).astype(np.float32)
        sf.write(audio_path, audio, SAMPLING_RATE)

        # extract mel spec
        mel = np.random.rand(80, 100).astype(np.float32).T
        np.save(MEL_SPEC_DIRECTORY.joinpath(f'{name}.npy'), mel)

        # get speaker embedding
        speaker_embedding = speaker_embedding_lookup.get(audio_file.filename)
        if speaker_embedding is None:
            try:
                speaker_embedding = _get_speaker_embedding(audio_path=audio_upload_path)
            except requests.exceptions.ConnectionError:
                return {'message': 'Speaker embedding server not available'}, HTTPStatus.INTERNAL_SERVER_ERROR
        np.save(SPK_EMB_DIRECTORY.joinpath(f'{name}.npy'), speaker_embedding)

        # create file.list for extracting mouth frames
        with open(WORKING_DIRECTORY.joinpath(f'{TYPE}_file.list'), 'w') as f:
            f.write(f'{TYPE}/{name}\n')

        # extract mouth frames
        start_time = time.time()
        face_landmarks = get_face_landmarks(video_path=video_raw_path)[1]
        time_taken = round(time.time() - start_time, 2)
        logging.info(f'Extracting face landmarks took {time_taken} secs')
        if any([l is None for l in face_landmarks]):
            return {'message': 'Failed to detect face landmarks'}, HTTPStatus.BAD_REQUEST
        with open(video_landmarks_path, 'wb') as f:
            pickle.dump(face_landmarks, f)
        response = execute_request('MOUTH FRAMES', requests.post, 'http://127.0.0.1:5003/extract_mouth_frames', json={'root': str(WORKING_DIRECTORY)})
        if response.status_code != HTTPStatus.NO_CONTENT:
            return {'message': 'Failed to extract mouth frames'}, HTTPStatus.INTERNAL_SERVER_ERROR

        # create manifests
        create_manifests(SimpleNamespace(**{'type': TYPE, 'dataset_directory': WORKING_DIRECTORY, 'dict_path': None}))

        # extract speech units
        num_speech_units = int(video_duration * AUDIO_FRAME_RATE)
        speech_units = ['14'] * num_speech_units
        with LABEL_DIRECTORY.joinpath('test.unt').open('w') as f:
            f.write(f'{" ".join(speech_units)}\n')

        # run synthesis
        response = execute_request('SYNTHESIS', requests.post, 'http://127.0.0.1:5004/synthesise')
        if response.status_code != HTTPStatus.NO_CONTENT:
            return {'message': 'Failed to run synthesiser'}, HTTPStatus.INTERNAL_SERVER_ERROR

        # setup vocoder directory
        setup_vocoder_inference(SimpleNamespace(**{'type': TYPE, 'dataset_directory': WORKING_DIRECTORY, 'synthesis_directory': SYNTHESIS_DIRECTORY}))

        # run vocoder
        response = execute_request('VOCODER', requests.post, 'http://127.0.0.1:5005/vocoder')
        if response.status_code != HTTPStatus.NO_CONTENT:
            return {'message': 'Failed to run vocoder'}, HTTPStatus.INTERNAL_SERVER_ERROR

        # post-processing audio - denoise and normalise
        post_process_audio(
            audio_path=pred_audio_path,
            output_path=str(POST_PROCESSED_AUDIO_PATH),
            sr=SAMPLING_RATE
        )
        pred_audio_path = str(POST_PROCESSED_AUDIO_PATH)

        # overlay onto video
        overlay_audio(video_raw_path, pred_audio_path, video_upload_path)

        # browser video playback compatibility, h264 is pretty universal
        convert_video_codecs(
            input_video_path=video_upload_path,
            video_codec='libx264',
            audio_codec='aac',
            output_video_path=video_download_path
        )

        # get asr results - on by default
        run_asr = bool(int(request.args.get('asr', 1)))
        asr_preds = []
        if run_asr:
            start_time = time.time()
            asr_preds = asr.run(pred_audio_path)
            time_taken = round(time.time() - start_time, 2)
            logging.info(f'Whisper ASR took {time_taken} secs')
    
        # log results in the db
        with DB(DB_PATH) as cur:
            usage_id = str(uuid.uuid4())

            video_id = name

            model_id = cur.execute(f'SELECT id FROM model WHERE name=\'{checkpoint_id}\'').fetchone()[0]

            audio_id_row = cur.execute(f'SELECT id FROM audio WHERE name=\'{audio_file.filename}\'').fetchone()
            audio_id = audio_id_row[0] if audio_id_row else None

            asr_pred = asr_preds[0] if len(asr_preds) >= 1 else ''

            cur.execute('INSERT INTO usage (id, model_id, video_id, audio_id, date) values (?, ?, ?, ?, ?)', (usage_id, model_id, video_id, audio_id, datetime.now()))
            cur.execute('INSERT INTO asr_transcription (id, usage_id, transcription) values (?, ?, ?)', (str(uuid.uuid4()), usage_id, asr_pred))

        return {
            'video_url': url_for('static', filename=Path(video_download_path).name),
            'asr_predictions': asr_preds
        }

    # pre-run the model loading for extracting landmarks which takes a while on first run
    get_face_landmarks(video_path='datasets/example.mp4')

    logging.info('Ready for requests...')

    return app


def main(args):
    app = web_app(args)
    app.run('0.0.0.0', port=args.port, debug=args.debug, use_reloader=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_paths', type=lambda s: s.split(','))
    parser.add_argument('--web_client_run_asr', action='store_true')
    parser.add_argument('--port', type=int, default=5002)
    parser.add_argument('--debug', action='store_true')

    main(parser.parse_args())
