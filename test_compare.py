import argparse
from pathlib import Path

import numpy as np
import torch
from jiwer import wer as calculate_wer
from tqdm import tqdm

import config
from helpers import WhisperASR, get_viseme_distance, get_words_to_visemes_d, load_groundtruth_data, preprocess_audio


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    l2s_directory = Path(args.l2s_directory)
    sv2s_directory = Path(args.sv2s_directory)

    output_d_name = 'pred_asr'
    if args.denoise_and_normalise:
        output_d_name += '_denoised_and_normalised'

    l2s_preds_directory = l2s_directory.joinpath(output_d_name)
    l2s_preds_directory.mkdir(exist_ok=True)

    asr = WhisperASR(model='medium', device=device)
    gt_df = load_groundtruth_data(args.groundtruth_path)[0]

    words_to_visemes_d = get_words_to_visemes_d()

    l2s_audio_paths = list(l2s_directory.joinpath('pred_wav/test').glob('*.wav'))
    l2s_wers, sv2s_wers = [], []
    l2s_vdists, sv2s_vdists = [], []

    for l2s_audio_path in tqdm(l2s_audio_paths): 
        name = l2s_audio_path.stem
        l2s_audio_path = str(l2s_audio_path)

        l2s_preds_path = l2s_preds_directory.joinpath(f'{name}_whisper_asr_results.txt')
        if l2s_preds_path.exists():
            with l2s_preds_path.open('r') as f:
                l2s_preds = f.read().splitlines()
        else:
            if args.denoise_and_normalise:
                new_audio_path = '/tmp/preprocessed_audio.wav'
                preprocess_audio(
                    audio_path=l2s_audio_path, 
                    output_path=new_audio_path, 
                    sr=config.SAMPLING_RATE
                )
                l2s_audio_path = new_audio_path

            l2s_preds = asr.run(l2s_audio_path)
            if not l2s_preds: 
                continue
        
            with l2s_preds_path.open('w') as f:
                for pred in l2s_preds:
                    f.write(f'{pred}\n')

        sv2s_preds_path = sv2s_directory.joinpath(f'{name}/Whisper_asr_results.txt')
        if not sv2s_preds_path.exists():
            continue
        with sv2s_preds_path.open('r') as f:
            sv2s_preds = f.read().splitlines()

        gt = gt_df[gt_df['Video Name'] == name]['Phrase'].values[0]
        l2s_pred = l2s_preds[0]
        sv2s_pred = sv2s_preds[0]
        print(f'"{gt}" - "{l2s_pred}" - "{sv2s_pred}"')

        l2s_wers.append(calculate_wer(gt, l2s_pred))
        sv2s_wers.append(calculate_wer(gt, sv2s_pred))

        l2s_vdist, sv2s_vdist = None, None
        try:
            l2s_vdist = get_viseme_distance(gt, l2s_pred, words_to_visemes_d)
            sv2s_vdist = get_viseme_distance(gt, sv2s_pred, words_to_visemes_d)
        except KeyError:
            continue
        l2s_vdists.append(l2s_vdist)
        sv2s_vdists.append(sv2s_vdist)

    assert len(l2s_wers) == len(sv2s_wers)
    assert len(l2s_vdists) == len(sv2s_vdists)

    print('L2S:', np.mean(l2s_wers), np.mean(l2s_vdists))
    print('SV2S:', np.mean(sv2s_wers), np.mean(sv2s_vdists))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('l2s_directory')
    parser.add_argument('sv2s_directory')
    parser.add_argument('groundtruth_path')
    parser.add_argument('--denoise_and_normalise', action='store_true')

    main(parser.parse_args())
