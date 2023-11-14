import os
from getpass import getpass
from pathlib import Path

import torch

# redis
REDIS_HOST = '127.0.0.1'
REDIS_PORT = 6379
REDIS_FRAME_QUEUE = 'vsg_frame_queue'
REDIS_LANDMARK_QUEUE = 'vsg_landmark_queue'
REDIS_SERVICE_QUEUE = 'vsg_service_queue'
REDIS_WAIT_TIME = 0.01

# lip2speech params
FPS = 25
SAMPLING_RATE = 16000
FILTER_LENGTH = 640
HOP_LENGTH = 160
WIN_LENGTH = 640
NUM_MEL_CHANNELS = 80
MEL_FMIN = 0.0
MEL_FMAX = 8000.0
AUDIO_FRAME_RATE = 50
DIM_1, DIM_2 = 640, 480
MAX_VIDEO_DURATION = 24

# synthesis setup
DB_PATH = Path('server.db')
STATIC_PATH = Path('static')
WEB_STATIC_PATH = Path('web')
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

# general
REPOS_PATH = Path(__file__).parents[1]
FAIRSEQ_PATH = REPOS_PATH.joinpath('fairseq')
SV2S_PATH = REPOS_PATH.joinpath('sv2s')
RNNOISE_PATH = REPOS_PATH.joinpath('rnnoise/examples/rnnoise_demo')
for p in [FAIRSEQ_PATH, SV2S_PATH, RNNOISE_PATH]:
    assert p.exists(), f'{p} does not exist'
MAX_GPU_DURATION = int(os.environ.get('MAX_GPU_DURATION', 6))  # for the decoder
USING_GPU = torch.cuda.is_available()
ALIGN_MOUTH_PORT = 5003
DECODER_PORT = 5004
VOCODER_PORT = 5005
DECODER_CPU_PORT = 5006

# email
EMAIL_HOST = os.environ.get('EMAIL_HOST', 'send.one.com')
EMAIL_PORT = int(os.environ.get('EMAIL_PORT', 587))
EMAIL_USERNAME = os.environ.get('EMAIL_USERNAME') or input('\nEnter email username: ')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD') or getpass('\nEnter email password: ')
EMAIL_RECEIVERS = [EMAIL_USERNAME]
