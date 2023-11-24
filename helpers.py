import collections
import pickle
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import ffmpeg
import numpy as np

import config

sys.path.extend([str(config.REPOS_PATH), str(config.SV2S_PATH)])
from sv2s.asr import WhisperASR
from sv2s.denoise import denoise_audio
from sv2s.detector import filter_landmarks, get_face_landmarks as get_ibug_landmarks, get_mouth_frames, init_facial_detectors as init_ibug_facial_detectors
from sv2s.utils import convert_fps, convert_video_codecs, crop_video, get_fps, get_sample_rate, get_speaker_embedding, \
    get_video_duration, get_viseme_distance, get_words_to_visemes_d, load_groundtruth_data, overlay_audio, split_list

FFMPEG_PATH = 'ffmpeg'
FFMPEG_OPTIONS = '-hide_banner -loglevel error'

# ffmpeg commands
EXTRACT_AUDIO_COMMAND = f'{FFMPEG_PATH} {FFMPEG_OPTIONS} -i {{input_video_path}} -f wav -vn -y -ac 1 -ar {{sr}} {{output_audio_path}}'
NORMALISE_AUDIO_COMMAND = f'ffmpeg-normalize -f -q {{input_audio_path}} -o {{output_audio_path}} -ar {{sr}}'
PAD_AUDIO_START_COMMAND = f'{FFMPEG_PATH} {FFMPEG_OPTIONS} -y -i {{input_audio_path}} -af "adelay={{delay}}000|{{delay}}000" {{output_audio_path}}'  # pads audio with delay seconds of silence
PAD_AUDIO_END_COMMAND = f'{FFMPEG_PATH} {FFMPEG_OPTIONS} -y -i {{input_audio_path}} -af "apad=pad_dur={{delay}}" {{output_audio_path}}'
REMOVE_AUDIO_PAD_COMMAND = f'{FFMPEG_PATH} {FFMPEG_OPTIONS} -y -i {{input_audio_path}} -ss 00:00:{{delay}}.000 -acodec pcm_s16le {{output_audio_path}}'  # removes delay seconds of silence
MERGE_VIDEOS_COMMAND = f'{FFMPEG_PATH} {FFMPEG_OPTIONS} -f concat -safe 0 -i {{file_list_path}} -c copy {{output_video_path}}'
RESIZE_VIDEO_COMMAND = f'{FFMPEG_PATH} {FFMPEG_OPTIONS} -y -i {{input_video_path}} -vf scale="{{width}}:{{height}}" {{output_video_path}}'

INVALID_VIDEO_FORMATS = ['image2', 'tty', 'ico', 'gif', 'pipe']


def run_command(s):
    subprocess.run(s, shell=True)


def time_wrapper(f, *args, **kwargs):
    start_time = time.time()
    f_out = f(*args, **kwargs)
    time_taken = round(time.time() - start_time, 2)

    return f_out, time_taken


def queue_frame(redis_cache, frame_index, frame, face_close_up=True):
    # for streaming, assume face close-up
    item = (frame_index, frame, face_close_up)
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
        v = np.asarray(v, dtype=object)[sorted_indexes].tolist()

    num_frames = len(video_landmarks['landmarks'])
    assert num_frames == max_frames_lambda(), f'{num_frames} landmarks != {max_frames_lambda()} frames'

    # TODO: won't need to filter landmarks if using video because POI already found
    return filter_landmarks(video_landmarks), sorted_indexes, video_landmarks['bbox']


def get_landmarks(redis_cache, video_path, max_frames, face_close_up=True):
    # reset queues
    for q in [config.REDIS_VIDEO_QUEUE, config.REDIS_LANDMARK_QUEUE]:
        redis_cache.delete(q) 

    # queue the video
    item = (video_path, face_close_up)
    redis_cache.rpush(config.REDIS_VIDEO_QUEUE, pickle.dumps(item))

    return dequeue_landmarks(redis_cache=redis_cache, max_frames_lambda=lambda: max_frames)[0]


def frame_to_bytes(frame):
    return cv2.imencode('.jpg', frame)[1].tobytes()


def bytes_to_frame(encoded_frame):
    return cv2.imdecode(np.frombuffer(encoded_frame, dtype=np.uint8), cv2.IMREAD_COLOR)


def is_valid_file(file_path, select_stream='video'):
    try:
        ffprobe_info = ffmpeg.probe(file_path, select_streams=select_stream[0])
    except Exception:
        return False

    specific_streams = ffprobe_info['streams']

    if select_stream == 'video':
        format_name = ffprobe_info['format']['format_name']
        
        return len(specific_streams) == 1 and not any([f in format_name for f in INVALID_VIDEO_FORMATS])

    # audio
    all_streams = ffmpeg.probe(file_path)['streams']
    
    return len(specific_streams) == 1 and len(all_streams) < 2


def merge_videos(file_list_path, output_video_path):
    run_command(MERGE_VIDEOS_COMMAND.format(
        file_list_path=file_list_path,
        output_video_path=output_video_path
    ))


def _get_speaker_embedding(audio_path):
    # convert to correct wav format
    converted_audio_file = tempfile.NamedTemporaryFile(suffix='.wav')
    extract_audio(video_path=audio_path, audio_path=converted_audio_file.name)
    audio_path = converted_audio_file.name

    # extract speaker embedding
    speaker_embedding = get_speaker_embedding(audio_path=audio_path)
    speaker_embedding = np.asarray(speaker_embedding, dtype=np.float32)
    assert speaker_embedding.shape == (256,) and speaker_embedding.dtype == np.float32

    converted_audio_file.close()

    return speaker_embedding


def get_updated_dims(width, height):
    # return min dims of input width/height or config width/height depending on orientation
    # ensure the same aspect ratio is kept
    is_landscape = width > height
    max_width, max_height = (config.DIM_1, config.DIM_2) if is_landscape else (config.DIM_2, config.DIM_1)
    aspect_ratio = width / height

    if is_landscape:
        # keep width fixed
        if width > max_width:
            width = max_width
        
        height = int(width / aspect_ratio)
    else:
        # keep height fixed
        if height > max_height:
            height = max_height

        width = int(height * aspect_ratio)

    assert width <= max_width and height <= max_height, f'({width}, {max_width}) !<= ({height}, {max_height})'

    return int(width), int(height)


def show_frames(video_frames, face_landmarks, bboxes=None, video_path=None):
    if not bboxes:
        bboxes = [[]] * len(video_frames)

    if video_path:
        video_frames = list(get_video_frames(video_path=video_path))

    debug_frames = []
    for frame, landmarks, bbox in zip(video_frames, face_landmarks, bboxes):
        # draw landmarks
        for x, y in landmarks:
            frame = cv2.circle(frame, (int(x), int(y)), radius=0, color=(0, 0, 255), thickness=2)

        # draw bbox
        for left, top, right, bottom in bbox:
            left, top, right, bottom = int(left), int(top), int(right), int(bottom)
            x, y = left, top
            w = right - left
            h = bottom - top
            frame = cv2.rectangle(frame, (left, top), (x + w, y + h), (0, 255, 0), 2)

        debug_frames.append(frame)

        cv2.imshow('Frame', frame)
        cv2.waitKey(25)

    cv2.destroyAllWindows()

    save_video(debug_frames, '/tmp/landmarks_debug.mp4', fps=config.FPS, colour=True)


def resize_video_opencv(video_path, width, height, fps):
    new_frames = [cv2.resize(frame, (width, height)) for frame in get_video_frames(video_path=video_path)]

    save_video(new_frames, video_path, fps=fps, colour=True)


def resize_video(video_path, width, height, output_video_path='/tmp/video_resized.mp4'):
    run_command(RESIZE_VIDEO_COMMAND.format(
        input_video_path=video_path,
        width=width,
        height=height,
        output_video_path=output_video_path
    ))
    shutil.copyfile(output_video_path, video_path)


def get_video_size(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return width, height


def extract_audio(video_path, audio_path):
    run_command(EXTRACT_AUDIO_COMMAND.format(
        input_video_path=video_path,
        sr=config.SAMPLING_RATE,
        output_audio_path=audio_path
    ))


def save_video(frames, video_path, fps, colour):
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height), colour)
    for frame in frames:
        video_writer.write(frame.astype(np.uint8))

    video_writer.release()


def get_video_frames(video_path):
    video_reader = cv2.VideoCapture(video_path)
    while True: 
        success, frame = video_reader.read()
        if not success:
            break

        yield frame

    video_reader.release()


def get_num_video_frames(video_path):
    return len(list(get_video_frames(video_path=video_path)))


def pad_audio(audio_path, delay, end=False):
    # pad silence at the start of the audio, end is optional

    pad_start_audio_file = tempfile.NamedTemporaryFile(suffix='.wav')
    run_command(PAD_AUDIO_START_COMMAND.format(
        input_audio_path=audio_path,
        delay=delay,
        output_audio_path=pad_start_audio_file.name
    ))

    if end:
        pad_end_audio_file = tempfile.NamedTemporaryFile(suffix='.wav')
        run_command(PAD_AUDIO_END_COMMAND.format(
            input_audio_path=pad_start_audio_file.name,
            delay=delay,
            output_audio_path=pad_end_audio_file.name
        ))

        pad_start_audio_file.close()
        return pad_end_audio_file

    return pad_start_audio_file


def remove_audio_pad(audio_file, delay):
    stripped_audio_file = tempfile.NamedTemporaryFile(suffix='.wav')

    run_command(REMOVE_AUDIO_PAD_COMMAND.format(
        input_audio_path=audio_file.name,
        delay=delay,
        output_audio_path=stripped_audio_file.name
    ))

    return stripped_audio_file


def normalise_audio(audio_file, sr=16000):
    normalised_audio_file = tempfile.NamedTemporaryFile(suffix='.wav')

    run_command(NORMALISE_AUDIO_COMMAND.format(
        input_audio_path=audio_file.name,
        output_audio_path=normalised_audio_file.name,
        sr=sr
    ))

    return normalised_audio_file


def preprocess_audio(audio_path, output_path, sr=16000):
    """
    It was found that denoising and then normalising the audio produced louder/more background noise
        - the denoising doesn't work as well on softer audio
        - then the normalising just makes the noise louder

    Normalising and then denoising the audio removed more noise but the sound was only slightly louder
        - normalising first makes the denoising process better
        - normalise again for good measure because the denoising process can make the speaking fainter

    WARNING: Ensure ffmpeg and ffmpeg-normalize libraries are up-to-date 
    """
    if not config.RNNOISE_PATH.exists():
        raise Exception(f'Failed to preprocess audio: {config.RNNOISE_PATH} does not exist...')

    audio_file = tempfile.NamedTemporaryFile(suffix='.wav')
    audio_file.name = audio_path

    denoised_audio_file = tempfile.NamedTemporaryFile(suffix='.wav')

    normalised_1_audio_file = normalise_audio(audio_file, sr=sr)
    denoise_audio(str(config.RNNOISE_PATH), normalised_1_audio_file.name, denoised_audio_file.name)
    normalised_2_audio_file = normalise_audio(denoised_audio_file, sr=sr)

    for f in [audio_file, normalised_1_audio_file, denoised_audio_file]:
        f.close()
    
    assert get_sample_rate(audio_path=normalised_2_audio_file.name) == sr

    shutil.copyfile(normalised_2_audio_file.name, output_path)
    normalised_2_audio_file.close()
