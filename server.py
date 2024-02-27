import argparse
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
from flask import Flask, Response, g as app_context, redirect, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import HTTPException

import config
from create_dataset import manifests as create_manifests, vocoder as setup_vocoder_inference
from email_client import send_email
from db import DB
from helpers import WhisperASR, bytes_to_frame, convert_fps, convert_video_codecs, debug_video, dequeue_landmarks, get_fps, \
    get_landmarks, get_mouth_frames, get_num_video_frames, _get_speaker_embedding, get_updated_dims, get_video_frames, \
        get_video_size, is_valid_file, overlay_audio, preprocess_audio as post_process_audio, queue_frame, resize_video, save_video, \
            time_wrapper

# TODO: 
# return full sized video with audio overlayed from resized video
#   - can't do this atm because frames are removed if missed landmark detections
#   - would need to remove same frames in original video too for a/v synchronisation
# containerise vsg
# parallel requests (no semaphores) - becomes an issue when VSG service being used
# RAM very high when using vsg service
# fix semaphore locking on failed requests e.g. 500s
# RAM growing - memory leak in decoders

asr = WhisperASR(model='base.en', device=os.environ.get('WHISPER_DEVICE', 'cpu'))
sem = threading.Semaphore()
stream_sem = threading.Semaphore()
video_frames = []
frame_dims = None
landmark_queue_id = str(uuid.uuid4())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)


def execute_request(name, r, *args, **kwargs):
    try:
        response, time_taken = time_wrapper(r, *args, **kwargs)
        logging.info(f'Request [{name}] took {time_taken} secs')
    except Exception as e:
        logging.error(f'Request [{name}] failed: {e}')
        response = Response(status=HTTPStatus.INTERNAL_SERVER_ERROR)
    
    return response


def setup(args, args_path):
    assert args or args_path, 'Args required'

    if args_path:
        with open(args_path, 'r') as f:
            args = SimpleNamespace(**json.load(f))

    for k in ['default_audios', 'default_audio_id', 'web_client_run_asr', 'web_client_streaming', 'prod']:
        assert k in args.__dict__, f'"{k}" not in args'

    assert len(args.default_audios) > 0, 'Default audios required'
    assert any([args.default_audio_id == d['id'] for d in args.default_audios]), f'Default audio "{args.default_audio_id}" does not exist in mapping'

    # setup static and server directories
    for d in [config.INPUTS_PATH, config.STATIC_PATH]:
        d.mkdir(parents=True, exist_ok=True)

    assert config.DB_PATH.exists(), 'Database does not exist, run migrations...'

    return args


def create_app(args=None, args_path=None):
    args = setup(args, args_path)

    # move any audio files to static and pre-load speaker embeddings for them
    default_audio_id = args.default_audio_id
    speaker_embedding_lookup = {}
    default_audios_list = []
    for audio_d in args.default_audios:
        audio_id, audio_name, audio_path = audio_d['id'], audio_d['name'], audio_d['path']

        audio_path = Path(audio_path)
        assert audio_path.exists(), f'{audio_path} does not exist'

        static_audio_path = config.STATIC_PATH.joinpath(f'{audio_id}.wav')
        shutil.copyfile(audio_path, static_audio_path)  # will replace if destination exists

        try:
            speaker_embedding_lookup[audio_id] = _get_speaker_embedding(audio_path=str(static_audio_path))
        except requests.exceptions.ConnectionError:
            logging.warning('Speaker embedding server is down...')
        
        default_audios_list.append({
            'id': audio_id,
            'name': audio_name
        })

    # get synthesiser checkpoints
    response = execute_request('GET CHECKPOINTS', requests.get, f'http://127.0.0.1:{config.DECODER_PORT}/checkpoints')
    if response.status_code != HTTPStatus.OK:
        logging.warning('Synthesiser is down...')
        checkpoint_ids = ['base']
    else:
        checkpoint_ids = response.json()

    # record the models
    with DB(config.DB_PATH) as cur:
        for checkpoint_id in checkpoint_ids:
            try:
                cur.execute('INSERT INTO model (id, name) VALUES (?, ?)', (str(uuid.uuid4()), checkpoint_id))
            except sqlite3.IntegrityError:
                pass

    app = Flask(__name__, static_folder=str(config.STATIC_PATH))
    app.secret_key = str(uuid.uuid4())
    socketio = SocketIO(app, serializer='msgpack')
    cache = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT)

    def _synthesise(uid, checkpoint_id, audio_id, audio_file=None, face_landmarks=None, face_close_up=True, run_asr=True, log_result=True):
        # acquire lock
        sem.acquire()

        logging.info(f'Synthesising UID {uid}...')

        video_upload_path = str(config.INPUTS_PATH.joinpath(f'{uid}.mp4'))
        audio_upload_path = str(config.INPUTS_PATH.joinpath(f'{uid}.wav'))
        video_raw_path = str(config.VIDEO_RAW_DIRECTORY.joinpath(f'{uid}.mp4'))
        video_landmarks_path = str(config.LANDMARKS_DIRECTORY.joinpath(f'{uid}.pkl'))
        mouth_video_path = str(config.VIDEO_DIRECTORY.joinpath(f'{uid}.mp4'))
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
        fps = get_fps(video_path=video_upload_path)
        width, height = get_video_size(video_path=video_upload_path)
        video_dims = get_updated_dims(width=width, height=height)
        if video_dims != (width, height):
            logging.info(f'Resizing video with (w, h) from ({width}, {height}) to {video_dims}')
            _, time_taken = time_wrapper(resize_video, video_upload_path, *video_dims)
            logging.info(f'Resize video took {time_taken} secs')

        # convert fps if applicable
        if fps != config.FPS:
            _, time_taken = time_wrapper(convert_fps, input_video_path=video_upload_path, fps=config.FPS, output_video_path=video_raw_path)
            logging.info(f'Convert FPS took {time_taken} secs')
        else:
            shutil.copyfile(video_upload_path, video_raw_path)

        num_video_frames = get_num_video_frames(video_path=video_raw_path)
        video_duration = num_video_frames / config.FPS

        # check video duration
        if video_duration > config.MAX_VIDEO_DURATION:
            return {'message': f'Video too long, must be <= {config.MAX_VIDEO_DURATION} seconds'}, HTTPStatus.BAD_REQUEST

        # get speaker embedding, priority given to uploaded audio file
        if audio_file:
            logging.info('Getting speaker embedding from audio file...')
            try:
                speaker_embedding, time_taken = time_wrapper(_get_speaker_embedding, audio_path=audio_upload_path)
                logging.info(f'Extracting speaker embedding took {time_taken} secs')
            except requests.exceptions.ConnectionError:
                return {'message': 'Speaker embedding server not available'}, HTTPStatus.INTERNAL_SERVER_ERROR
        else:
            logging.info(f'Using speaker embedding from lookup - {audio_id}...')
            speaker_embedding = speaker_embedding_lookup.get(audio_id)  # use default or supplied audio id
            if speaker_embedding is None:
                return {'message': f'Default audio id {audio_id} does not exist'}, HTTPStatus.BAD_REQUEST
        np.save(config.SPK_EMB_DIRECTORY.joinpath(f'{uid}.npy'), speaker_embedding)

        # create file.list for extracting mouth frames
        with open(config.WORKING_DIRECTORY.joinpath(f'{config.TYPE}_file.list'), 'w') as f:
            f.write(f'{config.TYPE}/{uid}\n')

        # get face landmarks
        # NOTE: if multiple people in the frame, POI is decided by maximum bb in the frame
        if face_landmarks is None:
            logging.info(f'Extracting face landmarks with face close-up: {face_close_up}')
            face_landmarks, time_taken = time_wrapper(get_landmarks, redis_cache=cache, video_path=video_raw_path, max_frames=num_video_frames, face_close_up=face_close_up)
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
            percentage_success = (num_valid_face_landmarks / num_video_frames) * 100
            logging.info(f'Failed to detect landmarks in some frames, excluding them: {percentage_success:.1f}% success')
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
        face_landmarks = [np.asarray(l) for l in face_landmarks]
        mouth_frames, time_taken = time_wrapper(get_mouth_frames, video_path=video_raw_path, landmarks=face_landmarks, greyscale=False)
        logging.info(f'Extracting mouth frames took {time_taken} secs')
        if mouth_frames is None:
            return {'message': 'Failed to extract mouth frames'}, HTTPStatus.INTERNAL_SERVER_ERROR
        save_video(mouth_frames, mouth_video_path, fps=config.FPS, colour=True)
        response = execute_request('COUNT FRAMES', requests.post, f'http://127.0.0.1:{config.ALIGN_MOUTH_PORT}/extract_mouth_frames', json={'root': str(config.WORKING_DIRECTORY)})
        if response.status_code != HTTPStatus.NO_CONTENT:
            return {'message': 'Failed to count frames'}, HTTPStatus.INTERNAL_SERVER_ERROR

        # create manifests
        create_manifests(SimpleNamespace(**{'type': config.TYPE, 'dataset_directory': config.WORKING_DIRECTORY, 'dict_path': None}))

        # extract speech units
        num_speech_units = int(video_duration * config.AUDIO_FRAME_RATE)
        speech_units = ['14'] * num_speech_units
        with config.LABEL_DIRECTORY.joinpath('test.unt').open('w') as f:
            f.write(f'{" ".join(speech_units)}\n')

        # run synthesis
        decoder_type, decoder_port = ('CPU', config.DECODER_CPU_PORT) if video_duration > config.MAX_GPU_DURATION else ('GPU', config.DECODER_PORT)
        response = execute_request(f'SYNTHESIS WITH {decoder_type}', requests.post, f'http://127.0.0.1:{decoder_port}/synthesise')
        if response.status_code != HTTPStatus.NO_CONTENT:
            return {'message': 'Failed to run synthesiser'}, HTTPStatus.INTERNAL_SERVER_ERROR

        # setup vocoder directory
        setup_vocoder_inference(SimpleNamespace(**{
            'type': config.TYPE, 
            'dataset_directory': config.WORKING_DIRECTORY, 
            'synthesis_directory': config.SYNTHESIS_DIRECTORY,
            'text_labels': False
        }))

        # run vocoder
        response = execute_request('VOCODER', requests.post, f'http://127.0.0.1:{config.VOCODER_PORT}/vocoder')
        if response.status_code != HTTPStatus.NO_CONTENT:
            return {'message': 'Failed to run vocoder'}, HTTPStatus.INTERNAL_SERVER_ERROR

        # overlay landmarks onto video
        if config.DEBUG:
            debug_video(
                video_frames=None,
                face_landmarks=face_landmarks,
                video_path=video_raw_path,
                save_path=video_raw_path
            )

        # post-processing audio - denoise and normalise
        _, time_taken = time_wrapper(post_process_audio,
            audio_path=pred_audio_path,
            output_path=str(config.POST_PROCESSED_AUDIO_PATH),
            sr=config.SAMPLING_RATE
        )
        logging.info(f'Post-processing audio took {time_taken} secs')
        pred_audio_path = str(config.POST_PROCESSED_AUDIO_PATH)

        # overlay onto video
        _, time_taken = time_wrapper(overlay_audio, video_raw_path, pred_audio_path, video_upload_path)
        logging.info(f'Overlay audio took {time_taken} secs')

        # browser video playback compatibility, h264 is pretty universal
        _, time_taken = time_wrapper(convert_video_codecs,
            input_video_path=video_upload_path,
            video_codec='libx264',
            audio_codec='aac',
            output_video_path=video_download_path
        )
        logging.info(f'Convert codecs took {time_taken} secs')

        # TODO: always run asr - run in background?
        # get asr results - on by default
        asr_preds = []
        if run_asr:
            asr_preds, time_taken = time_wrapper(asr.run, pred_audio_path)
            logging.info(f'Whisper ASR took {time_taken} secs')

        # log results in the db, optional for VSG service
        if log_result:
            with DB(config.DB_PATH) as cur:
                usage_id = str(uuid.uuid4())
                video_id = uid
                model_id = cur.execute(f'SELECT id FROM model WHERE name=\'{checkpoint_id}\'').fetchone()[0]
                cur.execute('INSERT INTO usage (id, model_id, video_id, audio_id, date) values (?, ?, ?, ?, ?)', (usage_id, model_id, video_id, None, datetime.now()))
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
            if config.DEBUG:
                logging.info(f'Resizing frame with (w, h) from ({width}, {height}) to {frame_dims}')
            frame = cv2.resize(frame, frame_dims)  # (width, height)

        queue_frame(redis_cache=cache, landmark_queue_id=landmark_queue_id, frame_index=frame_index, frame=frame)
        video_frames.append(frame)

        emit('response', f'frame {len(video_frames)} received')

    @socketio.on('end_stream')
    def end_stream(checkpoint_id, audio_id, audio_stream, run_asr):
        global video_frames

        logging.info('Client end stream; running synthesis...')

        uid = str(uuid.uuid4())
        video_upload_path = str(config.INPUTS_PATH.joinpath(f'{uid}.mp4'))
        audio_upload_path = str(config.INPUTS_PATH.joinpath(f'{uid}.wav'))

        # NOTE: audio_stream is bytes object
        audio_file = None
        if audio_stream:
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav')
            with open(temp_file.name, 'wb') as f:
                f.write(audio_stream)

            audio_file = FileStorage(stream=temp_file)
            audio_file.save(audio_upload_path)
            temp_file.close()

        # max frames keeps growing as lambda is executed constantly in loop
        face_landmarks, sorted_indexes, _ = dequeue_landmarks(redis_cache=cache, landmark_queue_id=landmark_queue_id, max_frames_lambda=lambda: len(video_frames))

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
            audio_id=audio_id if audio_id else default_audio_id,
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
            'checkpoint_ids': checkpoint_ids,
            'default_audios': default_audios_list,
            'web_client_run_asr': int(args.web_client_run_asr),
            'web_client_streaming': args.web_client_streaming
        })

    @app.post('/synthesise')
    def synthesise():
        uid = str(uuid.uuid4())
        video_upload_path = str(config.INPUTS_PATH.joinpath(f'{uid}.mp4'))
        audio_upload_path = str(config.INPUTS_PATH.joinpath(f'{uid}.wav'))

        # required video file
        video_file = request.files['video']
        _, time_taken = time_wrapper(video_file.save, video_upload_path)
        logging.info(f'Video save took {time_taken} secs')

        # optional audio file for speaker embedding
        audio_file = request.files.get('audio')
        if audio_file:
            audio_file.save(audio_upload_path)

        response = _synthesise(
            uid=uid,
            checkpoint_id=request.args.get('cid', 'base'),
            audio_id=request.args.get('aid', default_audio_id),  # optional audio id, fallback to default
            audio_file=audio_file,
            face_close_up=bool(int(request.args.get('close_up', 1))),
            run_asr=bool(int(request.args.get('asr', 1))),
            log_result=bool(int(request.args.get('log', 1)))
        )

        # release lock
        sem.release()

        return response

    @app.get('/audios')
    def get_audios():
        return default_audios_list, HTTPStatus.OK

    @app.get('/video/<video_id>')
    def get_video(video_id):
        return redirect(f'/static/{video_id}.mp4')

    @app.get('/audio/<audio_id>')
    def get_audio(audio_id):
        return redirect(f'/static/{audio_id}.wav')

    @app.get('/vsg')
    def vsg():
        return render_template('vsg.html', **{
            'default_audios': default_audios_list,
        })
    
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
            
            audio_path = None
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
            'email': email,
            'video_path': video_path,
            'audio_path': audio_path
        }))

        # notify liopa personnel of request
        if config.EMAIL_USERNAME:
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

    socketio.run(app, host='0.0.0.0', port=args.port, debug=config.DEBUG, use_reloader=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('args_path')
    parser.add_argument('--port', type=int, default=5002)

    cl_args = parser.parse_args()
    args = setup(cl_args, cl_args.args_path)
    args.port = cl_args.port

    create_app(args)
