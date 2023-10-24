import argparse
import collections
import json
import logging
import pickle
import shutil
import sqlite3
import subprocess
import sys
import tempfile
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
import redis
import requests
import soundfile as sf
import torch
from flask import Flask, Response, g as app_context, redirect, render_template, request
from flask_socketio import SocketIO, emit
from werkzeug.datastructures import FileStorage

import config
from db import DB
from create_dataset import manifests as create_manifests, vocoder as setup_vocoder_inference
sys.path.append('/home/domhnall/Repos/sv2s')
from asr import WhisperASR
from audio_utils import preprocess_audio as post_process_audio
from detector import filter_landmarks
from face_landmarks_server import bytes_to_frame
from utils import convert_fps, convert_video_codecs, get_fps, get_speaker_embedding, get_video_frames, overlay_audio

device = torch.device('cuda:0' if config.USING_GPU else 'cpu')
asr = WhisperASR(model='base.en', device=device)
sem = threading.Semaphore()
stream_sem = threading.Semaphore()
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
video_frames = []
frame_dims = None


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


def save_video(frames, video_path, fps, colour):
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height), colour)
    for frame in frames:
        video_writer.write(frame.astype(np.uint8))

    video_writer.release()


def queue_encoded_frame(redis_cache, encoded_frame):
    redis_cache.rpush(config.REDIS_FRAME_QUEUE, encoded_frame)


def dequeue_landmarks(redis_cache, max_frames_lambda):
    video_landmarks = collections.defaultdict(list)
    i = 0
    while i < max_frames_lambda():
        item = redis_cache.lpop(config.REDIS_LANDMARK_QUEUE)
        if not item:
            continue

        frame_landmarks = json.loads(item)

        video_landmarks['bbox'].extend(frame_landmarks['bbox'])
        video_landmarks['landmarks'].extend(frame_landmarks['landmarks'])
        video_landmarks['landmarks_scores'].extend(frame_landmarks['landmarks_scores'])

        i += 1

    num_frames = len(video_landmarks['landmarks'])
    assert num_frames == max_frames_lambda(), f'{num_frames} landmarks != {max_frames_lambda()} frames'

    return video_landmarks


def frame_to_bytes(frame):
    return cv2.imencode('.jpg', frame)[1].tobytes()


def get_landmarks(redis_cache, video_path):
    video_frames = get_video_frames(video_path=video_path)

    for frame in video_frames:
        queue_encoded_frame(redis_cache=redis_cache, encoded_frame=frame_to_bytes(frame))

    max_frames = len(video_frames)

    return filter_landmarks(dequeue_landmarks(redis_cache=redis_cache, max_frames_lambda=lambda: max_frames))


def show_frames(video_frames, face_landmarks):
    for frame, landmarks in zip(video_frames, face_landmarks):
        for x, y in landmarks:
            frame = cv2.circle(frame, (int(x), int(y)), radius=0, color=(0, 0, 255), thickness=3)
        cv2.imshow('Frame', frame)
        cv2.waitKey(25)

    cv2.destroyAllWindows()


def get_updated_dims(width, height):
    is_landscape = width > height
    max_width, max_height = (config.DIM_1, config.DIM_2) if is_landscape else (config.DIM_2, config.DIM_1)

    if width > max_width or height > max_height:
        return max_width, max_height

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


def setup(args, args_path):
    assert args or args_path, 'Args required'

    if args_path:
        with open(args_path, 'r') as f:
            args = SimpleNamespace(**json.load(f))

    assert 'audio_paths' in args.__dict__
    args.audio_paths = [p for p in args.audio_paths if p]
    assert len(args.audio_paths) > 0, f'Audio paths required'
    assert len(args.audio_paths) == len(set(args.audio_paths)), f'Non-unique audio paths'
    assert 'web_client_run_asr' in args.__dict__
    assert 'prod' in args.__dict__

    # setup static and server directories
    for d in [config.INPUTS_PATH, config.STATIC_PATH]:
        d.mkdir(parents=True, exist_ok=True)

    assert config.DB_PATH.exists(), 'Database does not exist, run migrations...'

    return args


def create_app(args=None, args_path=None):
    args = setup(args, args_path)

    # move any audio files to static and pre-load speaker embeddings for them
    for i, audio_path in enumerate(args.audio_paths):
        audio_path = Path(audio_path)
        assert audio_path.exists(), f'{audio_path} does not exist'
        new_audio_path = config.STATIC_PATH.joinpath(f'speaker_audio_{i + 1}.wav')
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
        checkpoint_ids = ['base']
    else:
        checkpoint_ids = response.json()

    # record the models and speaker audios
    with DB(config.DB_PATH) as cur:
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

    app = Flask(__name__, static_folder=str(config.STATIC_PATH))
    app.secret_key = str(uuid.uuid4())
    socketio = SocketIO(app, serializer='msgpack')
    cache = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT)

    def _synthesise(name, checkpoint_id, audio_file, face_landmarks=None, run_asr=True):
        # acquire lock
        sem.acquire()

        video_upload_path = str(config.INPUTS_PATH.joinpath(f'{name}.mp4'))
        audio_upload_path = str(config.INPUTS_PATH.joinpath(f'{name}.wav'))
        video_raw_path = str(config.VIDEO_RAW_DIRECTORY.joinpath(f'{name}.mp4'))
        video_landmarks_path = str(config.LANDMARKS_DIRECTORY.joinpath(f'{name}.pkl'))
        audio_path = str(config.AUDIO_DIRECTORY.joinpath(f'{name}.wav'))
        pred_audio_path = str(config.VOCODER_DIRECTORY.joinpath(f'pred_wav/{config.TYPE}/{name}.wav'))
        video_download_path = str(config.STATIC_PATH.joinpath(f'{name}.mp4'))

        # setup directory
        if config.WORKING_DIRECTORY.exists():
            shutil.rmtree(config.WORKING_DIRECTORY)
        for d in [config.WORKING_DIRECTORY, config.VIDEO_RAW_DIRECTORY, config.AUDIO_DIRECTORY, config.VIDEO_DIRECTORY, 
                config.MEL_SPEC_DIRECTORY, config.SPK_EMB_DIRECTORY, config.LANDMARKS_DIRECTORY, config.LABEL_DIRECTORY]:
            d.mkdir(parents=True)

        # convert size if applicable to prevent GPU memory overload
        if config.USING_GPU:
            width, height = get_video_size(video_path=video_upload_path)
            video_dims = get_updated_dims(width=width, height=height)
            if video_dims != (width, height):
                logging.info(f'Resizing video with (w, h) from ({width}, {height}) to {video_dims}')
                resize_video(video_upload_path, *video_dims)

        # convert fps if applicable
        if get_fps(video_path=video_upload_path) != config.FPS:
            convert_fps(input_video_path=video_upload_path, fps=config.FPS, output_video_path=video_raw_path)
        else:
            shutil.copyfile(video_upload_path, video_raw_path)

        num_video_frames = len(get_video_frames(video_path=video_raw_path))
        video_duration = num_video_frames / config.FPS

        # check video duration
        if config.USING_GPU and video_duration > config.MAX_GPU_DURATION:
            return {'message': f'Video too long, must be <= {config.MAX_GPU_DURATION} seconds'}, HTTPStatus.BAD_REQUEST

        # optional model checkpoint id
        response = execute_request('LOAD CHECKPOINT', requests.post, 'http://127.0.0.1:5004/load_checkpoint', json={'checkpoint_id': checkpoint_id})
        if response.status_code != HTTPStatus.NO_CONTENT:
            return {'message': 'Failed to load checkpoint'}, HTTPStatus.INTERNAL_SERVER_ERROR

        # extract audio
        num_audio_frames = int(video_duration * config.SAMPLING_RATE)
        audio = np.random.rand(num_audio_frames).astype(np.float32)
        sf.write(audio_path, audio, config.SAMPLING_RATE)

        # extract mel spec
        mel = np.random.rand(80, 100).astype(np.float32).T
        np.save(config.MEL_SPEC_DIRECTORY.joinpath(f'{name}.npy'), mel)

        # get speaker embedding
        logging.info(f'Getting speaker embedding from "{audio_file.filename}"')
        speaker_embedding = speaker_embedding_lookup.get(audio_file.filename)
        if speaker_embedding is None:
            logging.info('Preloaded embedding doesn\'t exist, retrieving new embedding...')
            try:
                speaker_embedding = _get_speaker_embedding(audio_path=audio_upload_path)
            except requests.exceptions.ConnectionError:
                return {'message': 'Speaker embedding server not available'}, HTTPStatus.INTERNAL_SERVER_ERROR
        np.save(config.SPK_EMB_DIRECTORY.joinpath(f'{name}.npy'), speaker_embedding)

        # create file.list for extracting mouth frames
        with open(config.WORKING_DIRECTORY.joinpath(f'{config.TYPE}_file.list'), 'w') as f:
            f.write(f'{config.TYPE}/{name}\n')

        # get face landmarks
        start_time = time.time()
        face_landmarks = face_landmarks if face_landmarks is not None else get_landmarks(redis_cache=cache, video_path=video_raw_path)
        time_taken = round(time.time() - start_time, 2)
        logging.info(f'Extracting face landmarks took {time_taken} secs')
        if any([l is None for l in face_landmarks]):
            return {'message': 'Failed to detect face landmarks'}, HTTPStatus.BAD_REQUEST
        with open(video_landmarks_path, 'wb') as f:
            pickle.dump(face_landmarks, f)

        # extract mouth frames
        response = execute_request('MOUTH FRAMES', requests.post, 'http://127.0.0.1:5003/extract_mouth_frames', json={'root': str(config.WORKING_DIRECTORY)})
        if response.status_code != HTTPStatus.NO_CONTENT:
            return {'message': 'Failed to extract mouth frames'}, HTTPStatus.INTERNAL_SERVER_ERROR

        # create manifests
        create_manifests(SimpleNamespace(**{'type': config.TYPE, 'dataset_directory': config.WORKING_DIRECTORY, 'dict_path': None}))

        # extract speech units
        num_speech_units = int(video_duration * config.AUDIO_FRAME_RATE)
        speech_units = ['14'] * num_speech_units
        with config.LABEL_DIRECTORY.joinpath('test.unt').open('w') as f:
            f.write(f'{" ".join(speech_units)}\n')

        # run synthesis
        response = execute_request('SYNTHESIS', requests.post, 'http://127.0.0.1:5004/synthesise')
        if response.status_code != HTTPStatus.NO_CONTENT:
            return {'message': 'Failed to run synthesiser'}, HTTPStatus.INTERNAL_SERVER_ERROR

        # setup vocoder directory
        setup_vocoder_inference(SimpleNamespace(**{'type': config.TYPE, 'dataset_directory': config.WORKING_DIRECTORY, 'synthesis_directory': config.SYNTHESIS_DIRECTORY}))

        # run vocoder
        response = execute_request('VOCODER', requests.post, 'http://127.0.0.1:5005/vocoder')
        if response.status_code != HTTPStatus.NO_CONTENT:
            return {'message': 'Failed to run vocoder'}, HTTPStatus.INTERNAL_SERVER_ERROR

        # post-processing audio - denoise and normalise
        post_process_audio(
            audio_path=pred_audio_path,
            output_path=str(config.POST_PROCESSED_AUDIO_PATH),
            sr=config.SAMPLING_RATE
        )
        pred_audio_path = str(config.POST_PROCESSED_AUDIO_PATH)

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
        asr_preds = []
        if run_asr:
            start_time = time.time()
            asr_preds = asr.run(pred_audio_path)
            time_taken = round(time.time() - start_time, 2)
            logging.info(f'Whisper ASR took {time_taken} secs')

        # log results in the db
        with DB(config.DB_PATH) as cur:
            usage_id = str(uuid.uuid4())
            video_id = name
            model_id = cur.execute(f'SELECT id FROM model WHERE name=\'{checkpoint_id}\'').fetchone()[0]
            audio_id_row = cur.execute(f'SELECT id FROM audio WHERE name=\'{audio_file.filename}\'').fetchone()
            audio_id = audio_id_row[0] if audio_id_row else None
            cur.execute('INSERT INTO usage (id, model_id, video_id, audio_id, date) values (?, ?, ?, ?, ?)', (usage_id, model_id, video_id, audio_id, datetime.now()))
            if asr_preds:
                cur.execute('INSERT INTO asr_transcription (id, usage_id, transcription) values (?, ?, ?)', (str(uuid.uuid4()), usage_id, asr_preds[0]))

        return {
            'videoId': name,
            'asrPredictions': asr_preds
        }, HTTPStatus.OK

    @socketio.on('connect')
    def connect():
        global video_frames, frame_dims

        stream_sem.acquire()

        video_frames = []
        frame_dims = None
        for q in [config.REDIS_FRAME_QUEUE, config.REDIS_LANDMARK_QUEUE]:
            cache.delete(q)

        logging.info('Client connected')

        emit('response', 'connected')

    @socketio.on('frame')
    def receive_frame(frame_index, encoded_frame):
        global video_frames, frame_dims

        frame = bytes_to_frame(encoded_frame)

        if config.USING_GPU:
            height, width = frame.shape[:2]
            
            if not frame_dims:
                frame_dims = get_updated_dims(width=width, height=height)
                logging.info(f'Retrieved updated dims (w, h): before ({width}, {height}), after {frame_dims}')

            if frame_dims != (width, height):
                frame = cv2.resize(frame, frame_dims)  # (width, height)
                encoded_frame = frame_to_bytes(frame)

        queue_encoded_frame(redis_cache=cache, encoded_frame=encoded_frame)
        video_frames.append([frame_index, frame])
        
        emit('response', f'frame {len(video_frames)} received')

    @socketio.on('end_stream')
    def end_stream(checkpoint_id, audio_stream, audio_name, run_asr):
        global video_frames

        logging.info('Client end stream; running synthesis...')

        name = str(uuid.uuid4())
        video_upload_path = str(config.INPUTS_PATH.joinpath(f'{name}.mp4'))
        audio_upload_path = str(config.INPUTS_PATH.joinpath(f'{name}.wav'))

        # NOTE: audio_stream is bytes object
        audio_file = FileStorage(filename='speaker_audio_1.wav')
        if audio_stream:
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav')
            with open(temp_file.name, 'wb') as f:
                f.write(audio_stream)

            audio_file = FileStorage(stream=temp_file)
            audio_file.save(audio_upload_path)
            temp_file.close()

            if audio_name:
                audio_file.filename = audio_name

        # max frames keeps growing as lambda is executed constantly in loop
        face_landmarks = dequeue_landmarks(redis_cache=cache, max_frames_lambda=lambda: len(video_frames))
        
        # TODO:
        #  fix error: server=127.0.0.1:5002//socket.io/ client=127.0.0.1:58606 socket shutdown error: [Errno 9] Bad file descriptor
        #  seems to be because of bad disconnect from the client side?
        #  "disconnect" is called on server side anyway - stream semaphore is released too

        # frames can arrive in a random order, need to sort them and landmarks accordingly
        # only then can you filter the landmarks
        assert len(video_frames) == len(face_landmarks['landmarks'])
        unsorted_frame_indexes, unsorted_frames = zip(*video_frames)
        sorted_indexes = np.argsort(unsorted_frame_indexes)
        _video_frames = np.asarray(unsorted_frames)[sorted_indexes]
        for v in face_landmarks.values():
            v = np.asarray(v)[sorted_indexes].tolist()
        face_landmarks = filter_landmarks(landmarks=face_landmarks)

        save_video(frames=_video_frames, video_path=video_upload_path, fps=config.FPS, colour=True)

        response = _synthesise(
            name=name,
            checkpoint_id=checkpoint_id,
            audio_file=audio_file,
            face_landmarks=face_landmarks,
            run_asr=bool(int(run_asr))
        )
        
        # release lock
        sem.release()

        emit('synthesise_response', response)

    @socketio.on('disconnect')
    def disconnect():
        stream_sem.release()

        logging.info('Client disconnected')

        emit('response', 'disconnected')

    @app.before_request
    def incoming_request():
        app_context.start_time = time.time()

    @app.after_request
    def outgoing_response(response):
        time_taken = round(time.time() - app_context.start_time, 2)
        logging.info(f'{request.method} {request.path} {dict(request.args)} {response.status_code} - {time_taken} secs')

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
        video_upload_path = str(config.INPUTS_PATH.joinpath(f'{name}.mp4'))
        audio_upload_path = str(config.INPUTS_PATH.joinpath(f'{name}.wav'))

        # required video file
        video_file = request.files['video']
        video_file.save(video_upload_path)

        # optional audio file for speaker embedding
        audio_file = FileStorage(filename='speaker_audio_1.wav')  # default speaker embedding
        if 'audio' in request.files:
            audio_file = request.files['audio']
            audio_file.save(audio_upload_path)

        response = _synthesise(
            name=name,
            checkpoint_id=request.form.get('checkpoint_id', 'base'),
            audio_file=audio_file,
            run_asr=bool(int(request.args.get('asr', 1)))
        )

        # release lock
        sem.release()

        return response

    @app.get('/video/<video_id>')
    def get_video(video_id):
        return redirect(f'/static/{video_id}.mp4')

    logging.info('Ready for requests...')

    if args.prod:
        return app

    socketio.run(app, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_paths', type=lambda s: s.split(','))
    parser.add_argument('--web_client_run_asr', action='store_true')
    parser.add_argument('--prod', action='store_true')
    parser.add_argument('--port', type=int, default=5002)
    parser.add_argument('--debug', action='store_true')

    create_app(parser.parse_args())
