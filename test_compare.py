import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from jiwer import wer as calculate_wer
from tqdm import tqdm

import config
from helpers import WhisperASR, expand_contractions, get_viseme_distance, get_words_to_visemes_d, load_groundtruth_data, preprocess_audio


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    l2s_directory = Path(args.l2s_directory)
    compare_to_directories = [Path(d) for d in args.compare_to_directories]

    output_d_name = 'pred_asr'
    if args.denoise_and_normalise:
        output_d_name += '_denoised_and_normalised'

    l2s_preds_directory = l2s_directory.joinpath(output_d_name)
    if l2s_preds_directory.exists() and args.redo:
        shutil.rmtree(l2s_preds_directory)
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
        l2s_pred = l2s_preds[0]

        # get preds to compare to - could be sv2s or l2s
        all_compare_preds = []
        for compare_to_directory in compare_to_directories:
            compare_preds_path = compare_to_directory.joinpath(f'{name}/Whisper_asr_results.txt')  # sv2s
            if not compare_preds_path.exists():
                compare_preds_path = compare_to_directory.joinpath(output_d_name).joinpath(f'{name}_whisper_asr_results.txt')  # l2s
                if not compare_preds_path.exists():
                    break
            with compare_preds_path.open('r') as f:
                compare_preds = f.read().splitlines()
            all_compare_preds.append(compare_preds[0])
        if len(all_compare_preds) != len(compare_to_directories):
            continue

        try:
            gt = gt_df[gt_df['Video Name'] == name]['Phrase'].values[0]
        except IndexError as e:
            print(f'{name} does not exist in groundtruth...')
            raise e

        # expand contractions
        gt = expand_contractions(text=gt)
        l2s_pred = expand_contractions(text=l2s_pred)
        all_compare_preds = [expand_contractions(text=p) for p in all_compare_preds]

        print(f'"{gt}" - "{l2s_pred}" - "{all_compare_preds}"')

        l2s_wers.append(calculate_wer(gt, l2s_pred))
        compare_wers.append([calculate_wer(gt, compare_pred) for compare_pred in all_compare_preds])

        try:
            l2s_vdist = get_viseme_distance(gt, l2s_pred, words_to_visemes_d, skip_words=args.vdist_skip_words)
            _compare_vdists = [get_viseme_distance(gt, compare_pred, words_to_visemes_d, skip_words=args.vdist_skip_words) for compare_pred in all_compare_preds]
        except KeyError:
            continue
        l2s_vdists.append(l2s_vdist)
        compare_vdists.append(_compare_vdists)

    assert len(l2s_wers) == len(compare_wers)
    assert len(l2s_vdists) == len(compare_vdists)

    l2s_wer, l2s_vdist = np.mean(l2s_wers), np.mean(l2s_vdists)
    print('Total tests:', len(l2s_wers))
    print(f'{args.l2s_directory} (original): WER {l2s_wer:.3f}, VDIST {l2s_vdist:.3f}')

    for i, compare_to_directory in enumerate(compare_to_directories):
        compare_wer = np.mean([x[i] for x in compare_wers])
        compare_vdist = np.mean([x[i] for x in compare_vdists])
        wer_diff, vdist_diff = abs(l2s_wer - compare_wer), abs(l2s_vdist - compare_vdist)
        print('------------------------------------------------------------------')
        print(f'{compare_to_directory}: WER {compare_wer:.3f}, VDIST {compare_vdist:.3f}')
        print(f'Diff to original: {wer_diff:.3f} WER, {vdist_diff:.3f} VDIST')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('l2s_directory')
    parser.add_argument('groundtruth_path')
    parser.add_argument('--compare_to_directories', type=lambda s: s.split(','), default=[])  # can be sv2s or l2s directories
    parser.add_argument('--denoise_and_normalise', action='store_true')
    parser.add_argument('--whisper_model', default='medium')
    parser.add_argument('--language', default='en')
    parser.add_argument('--vdist_skip_words', action='store_true')
    parser.add_argument('--redo', action='store_true')

    main(parser.parse_args())
