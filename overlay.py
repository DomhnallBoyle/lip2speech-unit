import argparse
import shutil
from pathlib import Path

from tqdm import tqdm

import config
from helpers import overlay_audio, preprocess_audio


def main(args):
    output_d_name = 'overlayed'
    if args.denoise_and_normalise:
        output_d_name += '_denoised_and_normalised'

    vocoder_results_directory = Path(args.vocoder_results_directory)
    videos_directory = Path(args.videos_directory)
    output_directory = vocoder_results_directory.joinpath(output_d_name)
    if output_directory.exists() and args.redo:
        shutil.rmtree(output_directory)
    output_directory.mkdir(exist_ok=True)

    for pred_audio_path in tqdm(vocoder_results_directory.joinpath('pred_wav/test').glob('*.wav')):
        video_name = f'{pred_audio_path.stem}.mp4'
        video_path = videos_directory.joinpath(video_name)
        assert video_path.exists(), f'{video_path} does not exist'

        output_video_path = output_directory.joinpath(video_name)
        if output_video_path.exists():
            continue

        audio_path = str(pred_audio_path)
        if args.denoise_and_normalise:
            new_audio_path = '/tmp/preprocessed_audio.wav'
            preprocess_audio(
                audio_path=audio_path, 
                output_path=new_audio_path, 
                sr=config.SAMPLING_RATE
            )
            audio_path = new_audio_path

        overlay_audio(
            input_video_path=str(video_path), 
            input_audio_path=audio_path, 
            output_video_path=str(output_video_path)
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('vocoder_results_directory')
    parser.add_argument('videos_directory')
    parser.add_argument('--denoise_and_normalise', action='store_true')
    parser.add_argument('--redo', action='store_true')

    main(parser.parse_args())
