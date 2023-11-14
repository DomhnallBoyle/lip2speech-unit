import argparse
import collections
import json
import logging
import os
import pickle
import shutil
import sqlite3
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
from flask import Flask, Response, g as app_context, redirect, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import HTTPException

import config
from create_dataset import manifests as create_manifests, vocoder as setup_vocoder_inference
from email_client import send_email
from db import DB
from helpers import WhisperASR, bytes_to_frame, convert_fps, convert_video_codecs, filter_landmarks, \
    get_fps, get_num_video_frames, _get_speaker_embedding, get_updated_dims, get_video_frames, get_video_size, \
        is_valid_file, overlay_audio, preprocess_audio as post_process_audio, resize_video, save_video

# TODO: 
# record in db the vsg service usages
# think about how stats will work, where will videos be saved etc
# need faster detector/predictor or ways to speed this up
    # use fast detector + ibug detector as backup for frames it couldn't get initially
    # can't use facenet-pytorch because it only returns 5 landmark points
    # need to wrap dlib with Dockerfile - use liopa/dlib
# ffmpeg-normalise seems to work with audio < 3 seconds now - speed up rnnoise
# use ibug face detectors/predictors for VSG service, dlib for SRAVI (speed reasons)?
# add UIDs to logs, :.2f in logs for time taken
# add wrapper function for timing things
# time detection through function call vs. redis queue

asr = WhisperASR(model='base.en', device='cpu')

sem = threading.Semaphore()
stream_sem = threading.Semaphore()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

speaker_embedding_lookup = {}
video_frames = []
frame_dims = None


def queue_frame(redis_cache, frame_index, frame):
    item = (frame_index, frame)
    redis_cache.rpush(config.REDIS_FRAME_QUEUE, pickle.dumps(item))


def dequeue_landmarks(redis_cache, max_frames_lambda):
    frame_indexes = []
    video_landmarks = collections.defaultdict(list)
    i = 0
    while i < max_frames_lambda():
        item = redis_cache.lpop(config.REDIS_LANDMARK_QUEUE)
        if not item:
            continue

        frame_index, frame_landmarks = pickle.loads(item)

        frame_indexes.append(frame_index)
        video_landmarks['bbox'].append(frame_landmarks['bbox'])
        video_landmarks['landmarks'].append(frame_landmarks['landmarks'])
        video_landmarks['landmarks_scores'].append(frame_landmarks['landmarks_scores'])

        i += 1

    # order landmarks correctly
    sorted_indexes = np.argsort(frame_indexes)
    for v in video_landmarks.values():
        v = np.asarray(v)[sorted_indexes].tolist()

    num_frames = len(video_landmarks['landmarks'])
    assert num_frames == max_frames_lambda(), f'{num_frames} landmarks != {max_frames_lambda()} frames'

    return filter_landmarks(video_landmarks), sorted_indexes


def get_landmarks(redis_cache, video_path):
    # reset queues
    for q in [config.REDIS_FRAME_QUEUE, config.REDIS_LANDMARK_QUEUE]:
        redis_cache.delete(q) 

    for i, frame in enumerate(get_video_frames(video_path=video_path)):
        queue_frame(redis_cache=redis_cache, frame_index=i, frame=frame)

    max_frames = i + 1

    return dequeue_landmarks(redis_cache=redis_cache, max_frames_lambda=lambda: max_frames)[0]


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
    assert 'web_client_streaming' in args.__dict__
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
    response = execute_request('GET CHECKPOINTS', requests.get, f'http://127.0.0.1:{config.DECODER_PORT}/checkpoints')
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

    def _synthesise(uid, checkpoint_id, audio_file, face_landmarks=None, run_asr=True):
        # acquire lock
        sem.acquire()

        video_upload_path = str(config.INPUTS_PATH.joinpath(f'{uid}.mp4'))
        audio_upload_path = str(config.INPUTS_PATH.joinpath(f'{uid}.wav'))
        video_raw_path = str(config.VIDEO_RAW_DIRECTORY.joinpath(f'{uid}.mp4'))
        video_landmarks_path = str(config.LANDMARKS_DIRECTORY.joinpath(f'{uid}.pkl'))
        audio_path = str(config.AUDIO_DIRECTORY.joinpath(f'{uid}.wav'))
        pred_audio_path = str(config.VOCODER_DIRECTORY.joinpath(f'pred_wav/{config.TYPE}/{uid}.wav'))
        video_download_path = str(config.STATIC_PATH.joinpath(f'{uid}.mp4'))

        # setup directory
        if config.WORKING_DIRECTORY.exists():
            shutil.rmtree(config.WORKING_DIRECTORY)
        for d in [config.WORKING_DIRECTORY, config.VIDEO_RAW_DIRECTORY, config.AUDIO_DIRECTORY, config.VIDEO_DIRECTORY, 
                config.MEL_SPEC_DIRECTORY, config.SPK_EMB_DIRECTORY, config.LANDMARKS_DIRECTORY, config.LABEL_DIRECTORY]:
            d.mkdir(parents=True)

        # optional model checkpoint id
        response = execute_request('LOAD CHECKPOINT', requests.post, f'http://127.0.0.1:{config.DECODER_PORT}/load_checkpoint', json={'checkpoint_id': checkpoint_id})
        if response.status_code != HTTPStatus.NO_CONTENT:
            return {'message': 'Failed to load checkpoint'}, HTTPStatus.INTERNAL_SERVER_ERROR

        # check uploaded files are valid
        for p, stream in zip([video_upload_path, audio_upload_path], ['video', 'audio']):
            if Path(p).exists() and not is_valid_file(p, select_stream=stream):
                return {'message': f'Uploaded {stream} file is not valid'}, HTTPStatus.BAD_REQUEST

        # convert size if applicable to prevent GPU memory overload
        width, height = get_video_size(video_path=video_upload_path)
        video_dims = get_updated_dims(width=width, height=height)
        if video_dims != (width, height):
            logging.info(f'Resizing video with (w, h) from ({width}, {height}) to {video_dims}')
            start_time = time.time()
            resize_video(video_upload_path, *video_dims)
            time_taken = round(time.time() - start_time, 2)
            logging.info(f'Resize video took {time_taken} secs')

        # convert fps if applicable
        if get_fps(video_path=video_upload_path) != config.FPS:
            start_time = time.time()
            convert_fps(input_video_path=video_upload_path, fps=config.FPS, output_video_path=video_raw_path)
            time_taken = round(time.time() - start_time, 2)
            logging.info(f'Convert FPS took {time_taken} secs')
        else:
            shutil.copyfile(video_upload_path, video_raw_path)

        num_video_frames = get_num_video_frames(video_path=video_raw_path)
        video_duration = num_video_frames / config.FPS

        # check video duration
        if video_duration > config.MAX_VIDEO_DURATION:
            return {'message': f'Video too long, must be <= {config.MAX_VIDEO_DURATION} seconds'}, HTTPStatus.BAD_REQUEST

        # get speaker embedding
        logging.info(f'Getting speaker embedding from "{audio_file.filename}"')
        speaker_embedding = speaker_embedding_lookup.get(audio_file.filename)
        if speaker_embedding is None:
            logging.info('Preloaded embedding doesn\'t exist, retrieving new embedding...')
            try:
                start_time = time.time()
                speaker_embedding = _get_speaker_embedding(audio_path=audio_upload_path)
                time_taken = round(time.time() - start_time, 2)
                logging.info(f'Extracting speaker embedding took {time_taken} secs')
            except requests.exceptions.ConnectionError:
                return {'message': 'Speaker embedding server not available'}, HTTPStatus.INTERNAL_SERVER_ERROR
        np.save(config.SPK_EMB_DIRECTORY.joinpath(f'{uid}.npy'), speaker_embedding)

        # create file.list for extracting mouth frames
        with open(config.WORKING_DIRECTORY.joinpath(f'{config.TYPE}_file.list'), 'w') as f:
            f.write(f'{config.TYPE}/{uid}\n')

        # get face landmarks
        # NOTE: if multiple people in the frame, POI is decided by maximum bb in the frame
        start_time = time.time()
        face_landmarks = face_landmarks if face_landmarks is not None else get_landmarks(redis_cache=cache, video_path=video_raw_path)
        time_taken = round(time.time() - start_time, 2)
        logging.info(f'Extracting face landmarks took {time_taken} secs')

        # deal with frames where face/landmarks not detected
        try:
            valid_face_landmark_indices, face_landmarks = zip(*[
                (i, l) for i, l in enumerate(face_landmarks) 
                if l is not None
            ])
        except ValueError:
            # unpack error - no faces detected in video
            return {'message': 'Failed to detect any faces'}, HTTPStatus.BAD_REQUEST
        
        face_landmarks = list(face_landmarks)
        num_valid_face_landmarks = len(valid_face_landmark_indices)

        # exclude non-detected frames
        if num_valid_face_landmarks != num_video_frames:
            logging.info(f'Failed to detect landmarks in some frames, excluding them')
            video_frames = list(get_video_frames(video_path=video_raw_path))
            video_frames = [video_frames[i] for i in valid_face_landmark_indices]
            num_video_frames = len(video_frames)
            video_duration = num_video_frames / config.FPS
            save_video(video_frames, video_raw_path, config.FPS, colour=True)

        assert num_valid_face_landmarks == num_video_frames

        # save face landmarks
        with open(video_landmarks_path, 'wb') as f:
            pickle.dump(face_landmarks, f)

        # extract audio
        num_audio_frames = int(video_duration * config.SAMPLING_RATE)
        audio = np.random.rand(num_audio_frames).astype(np.float32)
        sf.write(audio_path, audio, config.SAMPLING_RATE)

        # extract mel spec
        mel = np.random.rand(80, 100).astype(np.float32).T
        np.save(config.MEL_SPEC_DIRECTORY.joinpath(f'{uid}.npy'), mel)

        # extract mouth frames
        response = execute_request('MOUTH FRAMES', requests.post, f'http://127.0.0.1:{config.ALIGN_MOUTH_PORT}/extract_mouth_frames', json={'root': str(config.WORKING_DIRECTORY)})
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
        decoder_port = config.DECODER_CPU_PORT if video_duration > config.MAX_GPU_DURATION else config.DECODER_PORT
        response = execute_request(f'SYNTHESIS WITH PORT {decoder_port}', requests.post, f'http://127.0.0.1:{decoder_port}/synthesise')
        if response.status_code != HTTPStatus.NO_CONTENT:
            return {'message': 'Failed to run synthesiser'}, HTTPStatus.INTERNAL_SERVER_ERROR

        # setup vocoder directory
        setup_vocoder_inference(SimpleNamespace(**{'type': config.TYPE, 'dataset_directory': config.WORKING_DIRECTORY, 'synthesis_directory': config.SYNTHESIS_DIRECTORY}))

        # run vocoder
        response = execute_request('VOCODER', requests.post, f'http://127.0.0.1:{config.VOCODER_PORT}/vocoder')
        if response.status_code != HTTPStatus.NO_CONTENT:
            return {'message': 'Failed to run vocoder'}, HTTPStatus.INTERNAL_SERVER_ERROR

        # post-processing audio - denoise and normalise
        start_time = time.time()
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
        time_taken = round(time.time() - start_time, 2)
        logging.info(f'Post-processing took {time_taken} secs')

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
            video_id = uid
            model_id = cur.execute(f'SELECT id FROM model WHERE name=\'{checkpoint_id}\'').fetchone()[0]
            audio_id_row = cur.execute(f'SELECT id FROM audio WHERE name=\'{audio_file.filename}\'').fetchone()
            audio_id = audio_id_row[0] if audio_id_row else None
            cur.execute('INSERT INTO usage (id, model_id, video_id, audio_id, date) values (?, ?, ?, ?, ?)', (usage_id, model_id, video_id, audio_id, datetime.now()))
            if asr_preds:
                cur.execute('INSERT INTO asr_transcription (id, usage_id, transcription) values (?, ?, ?)', (str(uuid.uuid4()), usage_id, asr_preds[0]))

        return {
            'videoId': uid,
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
        height, width = frame.shape[:2]
        
        if not frame_dims:
            frame_dims = get_updated_dims(width=width, height=height)
            logging.info(f'Retrieved updated dims (w, h): before ({width}, {height}), after {frame_dims}')

        if frame_dims != (width, height):
            logging.info(f'Resizing frame with (w, h) from ({width}, {height}) to {frame_dims}')
            frame = cv2.resize(frame, frame_dims)  # (width, height)

        queue_frame(redis_cache=cache, frame_index=frame_index, frame=frame)
        video_frames.append(frame)

        emit('response', f'frame {len(video_frames)} received')

    @socketio.on('end_stream')
    def end_stream(checkpoint_id, audio_stream, audio_name, run_asr):
        global video_frames

        logging.info('Client end stream; running synthesis...')

        uid = str(uuid.uuid4())
        video_upload_path = str(config.INPUTS_PATH.joinpath(f'{uid}.mp4'))
        audio_upload_path = str(config.INPUTS_PATH.joinpath(f'{uid}.wav'))

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
        face_landmarks, sorted_indexes = dequeue_landmarks(redis_cache=cache, max_frames_lambda=lambda: len(video_frames))

        # TODO:
        #  fix error: server=127.0.0.1:5002//socket.io/ client=127.0.0.1:58606 socket shutdown error: [Errno 9] Bad file descriptor
        #  seems to be because of bad disconnect from the client side?
        #  "disconnect" is called on server side anyway - stream semaphore is released too

        # frames can arrive in a random order, need to sort them based on sorted landmark indices
        assert len(video_frames) == len(face_landmarks)
        unsorted_frames = video_frames
        _video_frames = np.asarray(unsorted_frames)[sorted_indexes]

        save_video(frames=_video_frames, video_path=video_upload_path, fps=config.FPS, colour=True)

        response = _synthesise(
            uid=uid,
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
    
    @app.errorhandler(Exception)
    def exception_handler(error):
        if isinstance(error, HTTPException):
            return error
        
        logging.error(f'Unhandled Exception: {traceback.format_exc()}')

        return {'message': 'Something went wrong...'}, HTTPStatus.INTERNAL_SERVER_ERROR

    @app.route('/cdn/<file_name>')
    def custom_static(file_name):
        return send_from_directory(config.WEB_STATIC_PATH, file_name)

    @app.get('/demo')
    def demo():
        return render_template('demo.html', **{
            'audio_paths': args.audio_paths,
            'checkpoint_ids': checkpoint_ids,
            'web_client_run_asr': int(args.web_client_run_asr),
            'web_client_streaming': args.web_client_streaming
        })

    @app.post('/synthesise')
    def synthesise():
        uid = str(uuid.uuid4())
        video_upload_path = str(config.INPUTS_PATH.joinpath(f'{uid}.mp4'))
        audio_upload_path = str(config.INPUTS_PATH.joinpath(f'{uid}.wav'))

        # required video file
        start_time = time.time()
        video_file = request.files['video']
        video_file.save(video_upload_path)
        time_taken = round(time.time() - start_time, 2)
        logging.info(f'Video save took {time_taken} secs')

        # optional audio file for speaker embedding
        audio_file = request.files.get('audio')
        if audio_file:
            audio_file.save(audio_upload_path)
        else:
            audio_file = FileStorage(filename='speaker_audio_1.wav')  # default speaker embedding

        response = _synthesise(
            uid=uid,
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

    @app.get('/vsg')
    def vsg():
        return render_template('vsg.html')
    
    @app.post('/dzupload')
    def upload():
        upload_id = request.args['id']
        file = request.files['file']
        upload_path = str(config.INPUTS_PATH.joinpath(f'{upload_id}_{file.filename}'))

        with open(upload_path, 'ab') as f:
            f.seek(int(request.form['dzchunkbyteoffset']))
            f.write(file.stream.read())

        current_chunk = int(request.form['dzchunkindex'])
        total_chunks = int(request.form['dztotalchunkcount'])

        if current_chunk + 1 == total_chunks:
            total_file_size = int(request.form['dztotalfilesize'])
            if os.path.getsize(upload_path) != total_file_size:
                return {'message': 'File size mismatch'}, HTTPStatus.INTERNAL_SERVER_ERROR

        return {'message': 'Chunk uploaded successfully'}, HTTPStatus.OK

    @app.post('/vsg/synthesise')
    def vsg_synthesise():
        # required email and upload id
        email = request.form['email']
        upload_id = request.form['uploadId']

        # get uploaded files
        uploaded_file_paths = [str(p) for p in config.INPUTS_PATH.glob(f'{upload_id}_*')]
        num_uploaded_files = len(uploaded_file_paths)

        # SCENARIOS:
        # - Uploaded 1 file - ensure is video file
        # - Uploaded 2 files - ensure 1 is video and the other is audio
        if num_uploaded_files == 1:
            video_path = uploaded_file_paths[0]
            if not is_valid_file(file_path=video_path, select_stream='video'):
                return {'message': 'Please upload a valid video file'}, HTTPStatus.BAD_REQUEST
            
            audio_path = 'static/speaker_audio_1.wav'
        elif num_uploaded_files == 2:
            video_path = None
            for i, p in enumerate(uploaded_file_paths):
                if is_valid_file(file_path=p, select_stream='video'):
                    video_path = p
                    break

            if video_path is None:
                return {'message': 'Please upload a valid video file'}, HTTPStatus.BAD_REQUEST
            
            audio_path = uploaded_file_paths[i - 1]  # -ve indexing, i = 0 (-1), i = 1 (0)
            if not is_valid_file(file_path=audio_path, select_stream='audio'):
                return {'message': 'The uploaded audio file is invalid'}, HTTPStatus.BAD_REQUEST
        else:
            return {'message': 'Too many files uploaded; upload a video and audio (optional)'}, HTTPStatus.BAD_REQUEST

        # push request to redis cache
        cache.rpush(config.REDIS_SERVICE_QUEUE, json.dumps({
            'url': request.host_url,
            'uid': upload_id,
            'video_path': video_path,
            'audio_path': audio_path
        }))

        # notify liopa personnel of request
        send_email(
            sender=config.EMAIL_USERNAME,
            receivers=config.EMAIL_RECEIVERS,
            subject=f'VSG Service Request - {upload_id}',
            content=f'{email} has requested to use the VSG service.\nThe uploaded video and audio paths are located in {video_path} and {audio_path} respectively.'
        )

        return {'message': 'Your request has been submitted successfully\nKeep checking your email for updates'}, HTTPStatus.OK

    logging.info('Ready for requests...')

    if args.prod:
        return app

    socketio.run(app, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_paths', type=lambda s: s.split(','))
    parser.add_argument('--web_client_run_asr', action='store_true')
    parser.add_argument('--web_client_streaming', action='store_true')
    parser.add_argument('--prod', action='store_true')
    parser.add_argument('--port', type=int, default=5002)
    parser.add_argument('--debug', action='store_true')

    create_app(parser.parse_args())
