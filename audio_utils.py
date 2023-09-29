import shutil
import subprocess
import sys
sys.path.append('/home/domhnall/Repos/sv2s')
import tempfile

from denoise import denoise_audio
from utils import get_sample_rate

FFMPEG_PATH = 'ffmpeg'
FFMPEG_OPTIONS = '-hide_banner -loglevel error'

# normalise commands
NORMALISE_AUDIO_COMMAND = f'ffmpeg-normalize -f -q {{input_audio_path}} -o {{output_audio_path}} -ar {{sr}}'
PAD_AUDIO_START_COMMAND = f'{FFMPEG_PATH} {FFMPEG_OPTIONS} -y -i {{input_audio_path}} -af "adelay={{delay}}000|{{delay}}000" {{output_audio_path}}'  # pads audio with delay seconds of silence
PAD_AUDIO_END_COMMAND = f'{FFMPEG_PATH} {FFMPEG_OPTIONS} -y -i {{input_audio_path}} -af "apad=pad_dur={{delay}}" {{output_audio_path}}'
REMOVE_AUDIO_PAD_COMMAND = f'{FFMPEG_PATH} {FFMPEG_OPTIONS} -y -i {{input_audio_path}} -ss 00:00:{{delay}}.000 -acodec pcm_s16le {{output_audio_path}}'  # removes delay seconds of silence


def pad_audio(audio_path, delay, end=False):
    # pad silence at the start of the audio, end is optional

    pad_start_audio_file = tempfile.NamedTemporaryFile(suffix='.wav')
    subprocess.call(PAD_AUDIO_START_COMMAND.format(
        input_audio_path=audio_path,
        delay=delay,
        output_audio_path=pad_start_audio_file.name
    ), shell=True)

    if end:
        pad_end_audio_file = tempfile.NamedTemporaryFile(suffix='.wav')
        subprocess.call(PAD_AUDIO_END_COMMAND.format(
            input_audio_path=pad_start_audio_file.name,
            delay=delay,
            output_audio_path=pad_end_audio_file.name
        ), shell=True)

        pad_start_audio_file.close()
        return pad_end_audio_file

    return pad_start_audio_file


def remove_audio_pad(audio_file, delay):
    stripped_audio_file = tempfile.NamedTemporaryFile(suffix='.wav')

    subprocess.call(REMOVE_AUDIO_PAD_COMMAND.format(
        input_audio_path=audio_file.name,
        delay=delay,
        output_audio_path=stripped_audio_file.name
    ), shell=True)

    return stripped_audio_file


def normalise_audio(audio_file, sr=16000):
    normalised_audio_file = tempfile.NamedTemporaryFile(suffix='.wav')

    subprocess.call(NORMALISE_AUDIO_COMMAND.format(
        input_audio_path=audio_file.name,
        output_audio_path=normalised_audio_file.name,
        sr=sr
    ), shell=True)

    return normalised_audio_file


def preprocess_audio(audio_path, output_path, delay=3, sr=16000):
    """
    It was found that denoising and then normalising the audio produced louder/more background noise
        - the denoising doesn't work as well on softer audio
        - then the normalising just makes the noise louder

    Normalising and then denoising the audio removed more noise but the sound was only slightly louder
        - normalising first makes the denoising process better
        - normalise again for good measure because the denoising process can make the speaking fainter

    Normalising requires audios >= 3 seconds, pad all with silence and remove after
    See: https://github.com/slhck/ffmpeg-normalize/issues/87
    """
    denoised_audio_file = tempfile.NamedTemporaryFile(suffix='.wav')

    # pad, normalise, denoise, normalise and strip
    padded_audio_file = pad_audio(audio_path, delay=delay)
    normalised_1_audio_file = normalise_audio(padded_audio_file, sr=sr)
    denoise_audio('/home/domhnall/Repos/rnnoise/examples/rnnoise_demo', normalised_1_audio_file.name, denoised_audio_file.name)
    normalised_2_audio_file = normalise_audio(denoised_audio_file, sr=sr)
    stripped_audio_file = remove_audio_pad(normalised_2_audio_file, delay=delay)

    for f in [padded_audio_file, normalised_1_audio_file,
              denoised_audio_file, normalised_2_audio_file]:
        f.close()

    assert get_sample_rate(audio_path=stripped_audio_file.name) == sr

    shutil.copyfile(stripped_audio_file.name, output_path)
    stripped_audio_file.close()  
