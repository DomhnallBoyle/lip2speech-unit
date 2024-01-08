import argparse
from pathlib import Path

import numpy as np
import torch
from jiwer import wer as calculate_wer
from tqdm import tqdm

import config
from helpers import WhisperASR, get_viseme_distance, get_words_to_visemes_d, load_groundtruth_data, preprocess_audio

# TODO: normalise contractions e.g. don't = do not

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    l2s_directory = Path(args.l2s_directory)
    compare_to_directory = Path(args.compare_to_directory) if args.compare_to_directory else None

    output_d_name = 'pred_asr'
    if args.denoise_and_normalise:
        output_d_name += '_denoised_and_normalised'

    l2s_preds_directory = l2s_directory.joinpath(output_d_name)
    l2s_preds_directory.mkdir(exist_ok=True)

    asr = WhisperASR(model=args.whisper_model, language=args.language, device=device)
    gt_df = load_groundtruth_data(args.groundtruth_path)[0]

    words_to_visemes_d = get_words_to_visemes_d(language=args.language)

    l2s_audio_paths = list(l2s_directory.joinpath('pred_wav/test').glob('*.wav'))
    l2s_wers, compare_wers = [], []
    l2s_vdists, compare_vdists = [], []

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

        # get preds to compare to - could be sv2s or l2s
        if compare_to_directory:
            compare_preds_path = compare_to_directory.joinpath(f'{name}/Whisper_asr_results.txt')  # sv2s
            if not compare_preds_path.exists():
                compare_preds_path = compare_to_directory.joinpath(output_d_name).joinpath(f'{name}_whisper_asr_results.txt')  # l2s
                if not compare_preds_path.exists():
                    continue
            with compare_preds_path.open('r') as f:
                compare_preds = f.read().splitlines()
        else:
            compare_preds = ['none']

        try:
            gt = gt_df[gt_df['Video Name'] == name]['Phrase'].values[0]
        except IndexError as e:
            print(f'{name} does not exist in groundtruth...')
            raise e
        l2s_pred = l2s_preds[0]
        compare_pred = compare_preds[0]
        print(f'"{gt}" - "{l2s_pred}" - "{compare_pred}"')

        l2s_wers.append(calculate_wer(gt, l2s_pred))
        compare_wers.append(calculate_wer(gt, compare_pred))

        l2s_vdist, compare_vdist = None, None
        try:
            l2s_vdist = get_viseme_distance(gt, l2s_pred, words_to_visemes_d, skip_words=args.vdist_skip_words)
            compare_vdist = get_viseme_distance(gt, compare_pred, words_to_visemes_d, skip_words=args.vdist_skip_words)
        except KeyError:
            continue
        l2s_vdists.append(l2s_vdist)
        compare_vdists.append(compare_vdist)

    assert len(l2s_wers) == len(compare_wers)
    assert len(l2s_vdists) == len(compare_vdists)

    l2s_wer, l2s_vdist = np.mean(l2s_wers), np.mean(l2s_vdists)
    compare_wer, compare_vdist = np.mean(compare_wers), np.mean(compare_vdists)
    wer_diff, vdist_diff = l2s_wer - compare_wer, l2s_vdist - compare_vdist

    print('Total tests:', len(l2s_wers))
    print(f'{args.l2s_directory}: WER {l2s_wer:.3f}, VDIST {l2s_vdist:.3f}')
    print(f'{args.compare_to_directory}: WER {compare_wer:.3f}, VDIST {compare_vdist:.3f}')
    print(f'Diff to original: {wer_diff:.3f} WER, {vdist_diff:.3f} VDIST')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('l2s_directory')
    parser.add_argument('groundtruth_path')
    parser.add_argument('--compare_to_directory')  # can be sv2s or l2s directory
    parser.add_argument('--denoise_and_normalise', action='store_true')
    parser.add_argument('--whisper_model', default='medium')
    parser.add_argument('--language', default='en')
    parser.add_argument('--vdist_skip_words', action='store_true')

    main(parser.parse_args())
