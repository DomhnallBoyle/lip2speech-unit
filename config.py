from pathlib import Path

import torch

REDIS_HOST = '127.0.0.1'
REDIS_PORT = 6379
REDIS_FRAME_QUEUE = 'vsg_frame_queue'
REDIS_LANDMARK_QUEUE = 'vsg_landmark_queue'

FPS = 25
SAMPLING_RATE = 16000
AUDIO_FRAME_RATE = 50
DIM_1, DIM_2 = 640, 480
MAX_GPU_DURATION = 6
USING_GPU = torch.cuda.is_available()
DB_PATH = Path('server.db')
STATIC_PATH = Path('static')
SERVER_PATH = Path('/tmp/server')
INPUTS_PATH = SERVER_PATH.joinpath('inputs')
POST_PROCESSED_AUDIO_PATH = SERVER_PATH.joinpath('post_processed_audio.wav')
WORKING_DIRECTORY = Path('/tmp/lip2speech')
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
