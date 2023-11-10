import argparse
import json
import multiprocessing
import pickle
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

import config

sys.path.append(str(config.FAIRSEQ_PATH))
from helpers import convert_fps, crop_video, extract_audio, get_face_landmarks, get_fps, get_num_video_frames, get_video_duration, init_facial_detectors, preprocess_audio, split_list
from examples.textless_nlp.gslm.unit2speech.tacotron2.layers import TacotronSTFT

stft = None

# TODO: 
#  fake audio, mel-spec and speech units for testing etc


def trim_video_to_duration(video_path, cropped_video_path='/tmp/video_cropped.mp4'):
    # fix duration issue where ffprobe duration doesn't match no. frames and fps
    num_video_frames = get_num_video_frames(video_path=video_path)
    video_duration = num_video_frames / config.FPS

    if video_duration != get_video_duration(video_path):
        crop_video(video_path, 0, video_duration, cropped_video_path)
        shutil.copyfile(cropped_video_path, video_path)

    return num_video_frames, video_duration 


def crop_to_random_duration(video_path, num_video_frames, video_duration, max_duration, cropped_video_path='/tmp/video_cropped.mp4'):
    if video_duration > max_duration:
        duration_num_frames = random.randint(1 * config.FPS, max_duration * config.FPS)
        frame_start_index = random.randint(0, (num_video_frames - duration_num_frames) - 1)
        frame_end_index = frame_start_index + duration_num_frames
        start_duration = frame_start_index / config.FPS
        end_duration = frame_end_index / config.FPS

        crop_video(video_path, start_duration, end_duration, cropped_video_path)
        shutil.copyfile(cropped_video_path, video_path)


def extract_mel_spec(audio_path):
    global stft
    if stft is None:
        stft = TacotronSTFT(filter_length=config.FILTER_LENGTH, hop_length=config.HOP_LENGTH, win_length=config.WIN_LENGTH,
                            n_mel_channels=config.NUM_MEL_CHANNELS, sampling_rate=config.SAMPLING_RATE, mel_fmin=config.MEL_FMIN,
                            mel_fmax=config.MEL_FMAX)

    audio, sr = sf.read(audio_path)
    assert sr == config.SAMPLING_RATE
    audio = torch.from_numpy(audio).unsqueeze(0).float()
    mel = stft.mel_spectrogram(audio)[0].numpy()
    assert mel.shape[0] == config.NUM_MEL_CHANNELS and mel.dtype == np.float32

    return mel.T


def init_process(process_index, args, sample_paths, already_processed, to_process):
    cropped_video_path = f'/tmp/video_cropped_{process_index}.mp4'
    preprocessed_audio_path = f'/tmp/preprocessed_audio_{process_index}.wav'

    if args.use_new_landmark_detector:
        init_facial_detectors()

    for sample_path in tqdm(sample_paths):
        if str(sample_path) in already_processed:
            continue

        if to_process and str(sample_path) not in to_process:
            continue

        # can't use original mouth frames because they were converted to 20 FPS
        sample = np.load(sample_path, allow_pickle=True)['sample']
        video_path, speaker_embedding = sample[0], sample[2]
        if args.replace_paths:
            for k, v in args.replace_paths.items():
                video_path = str(video_path).replace(k, v)

        video_path = Path(video_path)
        assert video_path.exists(), f'{video_path} does not exist'

        name = video_path.stem
        if args.use_unique_parent_name:
            name = f'{video_path.parents[0].name}_{name}'

        video_path = str(video_path)

        # copy raw video or convert fps if necessary
        raw_video_path = str(args.video_raw_directory.joinpath(f'{name}.mp4'))
        if get_fps(video_path) != config.FPS:
            convert_fps(video_path, config.FPS, raw_video_path)
        else:
            shutil.copyfile(video_path, raw_video_path)
        video_path = raw_video_path

        num_video_frames, video_duration = trim_video_to_duration(
            video_path=video_path, 
            cropped_video_path=cropped_video_path
        )
        if args.max_duration:
            crop_to_random_duration(
                video_path=video_path, 
                num_video_frames=num_video_frames, 
                video_duration=video_duration, 
                max_duration=args.max_duration,
                cropped_video_path=cropped_video_path
            )

        # extract audio
        audio_path = str(args.audio_directory.joinpath(f'{name}.wav'))
        extract_audio(video_path=video_path, audio_path=audio_path)

        if args.denoise_and_normalise:
            preprocess_audio(audio_path=audio_path, output_path=preprocessed_audio_path)
            shutil.copyfile(preprocessed_audio_path, audio_path)

        # create mel-spec
        mel = extract_mel_spec(audio_path=audio_path)
        mel_path = args.mel_spec_directory.joinpath(f'{name}.npy')
        np.save(mel_path, mel)

        # save speaker embedding
        speaker_embedding_path = args.spk_emb_directory.joinpath(f'{name}.npy')
        if args.speaker_embedding_path:
            speaker_embedding = np.load(args.speaker_embedding_path)
        assert speaker_embedding.shape == (256,) and speaker_embedding.dtype == np.float32
        np.save(speaker_embedding_path, speaker_embedding)

        # use better detector if applicable
        if args.use_new_landmark_detector:
            face_landmarks = get_face_landmarks(video_path=video_path)[1]
            with args.landmarks_directory.joinpath(f'{name}.pkl').open('wb') as f:
                pickle.dump(face_landmarks, f)

        with args.processed_path.open('a') as f:
            f.write(f'{sample_path}\n')


def init(args):
    dataset_directory = Path(args.dataset_directory)
    assert dataset_directory.exists()

    output_directory = Path(args.output_directory)
    if output_directory.exists() and args.redo:
        shutil.rmtree(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)

    video_raw_directory = output_directory.joinpath(f'{args.type}')
    audio_directory = output_directory.joinpath(f'audio/{args.type}')
    video_directory = output_directory.joinpath(f'video/{args.type}')
    mel_spec_directory = output_directory.joinpath(f'mel/{args.type}')
    spk_emb_directory = output_directory.joinpath(f'spk_emb/{args.type}')
    landmarks_directory = output_directory.joinpath(f'landmark/{args.type}')
    label_directory = output_directory.joinpath('label')
    for d in [video_raw_directory, audio_directory, video_directory, mel_spec_directory, spk_emb_directory, landmarks_directory, label_directory]:
        d.mkdir(exist_ok=True, parents=True)

    args.video_raw_directory = video_raw_directory
    args.audio_directory = audio_directory
    args.mel_spec_directory = mel_spec_directory
    args.spk_emb_directory = spk_emb_directory
    args.landmarks_directory = landmarks_directory
    args.processed_path = output_directory.joinpath(f'{args.type}_processed.txt')

    already_processed = set()
    if args.processed_path.exists():
        with args.processed_path.open('r') as f:
            already_processed = set(f.read().splitlines())

    sample_paths = list(dataset_directory.glob('*.npz'))
    if args.num_samples:
        random.shuffle(sample_paths)
        sample_paths = sample_paths[:args.num_samples]

    to_process = set()
    if args.samples_path:
        with open(args.samples_path, 'r') as f:
            to_process = set(f.read().splitlines())

    tasks = [[i, args, _sample_paths, already_processed, to_process]
             for i, _sample_paths in enumerate(split_list(sample_paths, args.num_processes))]
    with multiprocessing.Pool(processes=args.num_processes) as p:
        p.starmap(init_process, tasks)


def generate_file_list(args):
    dataset_directory = Path(args.dataset_directory)

    names = []
    for video_path in dataset_directory.joinpath(args.type).glob('*.mp4'):
        names.append(video_path.stem)

    # create file list
    with dataset_directory.joinpath(f'{args.type}_file.list').open('w') as f:
        for name in names:
            f.write(f'{args.type}/{name}\n')


def manifests(args):
    dataset_directory = Path(args.dataset_directory)
    assert dataset_directory.exists()
    root_path = str(dataset_directory.resolve())

    nframes_video_path = dataset_directory.joinpath('nframes.video.0')
    nframes_audio_path = dataset_directory.joinpath('nframes.audio.0')
    assert nframes_video_path.exists() and nframes_audio_path.exists()

    with nframes_video_path.open('r') as f:
        all_num_video_frames = f.read().splitlines()

    with nframes_audio_path.open('r') as f:
        all_num_audio_frames = f.read().splitlines()

    file_list_path = dataset_directory.joinpath(f'{args.type}_file.list')
    manifest_path = dataset_directory.joinpath(f'label/{args.type}.tsv')
    unit_manifest_path = dataset_directory.joinpath(f'{args.type}_unit_manifest.txt')
    with file_list_path.open('r') as f1, manifest_path.open('w') as f2, unit_manifest_path.open('w') as f3:
        f2.write(f'{root_path}\n')
        f3.write(f'{root_path}\n')

        for i, _id in enumerate(f1.read().splitlines()):
            video_path = f'video/{_id}.mp4'
            audio_path = f'audio/{_id}.wav'
            num_video_frames = all_num_video_frames[i]
            num_audio_frames = all_num_audio_frames[i]

            f2.write(f'{_id}\t{video_path}\t{audio_path}\t{num_video_frames}\t{num_audio_frames}\n')
            f3.write(f'{audio_path}\t{num_audio_frames}\n')

    if args.dict_path:
        shutil.copyfile(args.dict_path, dataset_directory.joinpath('label'))


def vocoder(args):
    dataset_directory = Path(args.dataset_directory)
    assert dataset_directory.exists()

    synthesis_directory = Path(args.synthesis_directory)
    assert synthesis_directory.exists()

    output_directory = synthesis_directory.joinpath('vocoder')
    if output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir()

    shutil.copytree(dataset_directory.joinpath('audio'), output_directory.joinpath('audio'))
    shutil.copytree(dataset_directory.joinpath('label'), output_directory.joinpath('label'))
    shutil.copytree(dataset_directory.joinpath('spk_emb'), output_directory.joinpath('spk_emb'))
    shutil.copytree(synthesis_directory.joinpath('pred_mel'), output_directory.joinpath('mel'))

    manifest_path = output_directory.joinpath(f'label/{args.type}.tsv')
    unt_path = output_directory.joinpath(f'label/{args.type}.unt')

    with manifest_path.open('r') as f:
        manifest = f.read().splitlines()
        manifest[0] = str(output_directory.resolve())

    with manifest_path.open('w') as f1, unt_path.open('w') as f2:
        f1.write(f'{manifest[0]}\n')

        for line in manifest[1:]:
            _id = line.split('\t')[0]
            pred_unit_path = synthesis_directory.joinpath(f'pred_unit/{_id}.txt')
            if not pred_unit_path.exists():
                continue
            with pred_unit_path.open('r') as f3:
                speech_units = f3.read().splitlines()[0]

            f1.write(f'{line}\n')
            f2.write(f'{speech_units}\n')


def main(args):
    f = {
        'init': init,
        'generate_file_list': generate_file_list,
        'manifests': manifests,
        'vocoder': vocoder
    }
    f[args.run_type](args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', choices=['train', 'val', 'test'])
    parser.add_argument('dataset_directory')

    sub_parser = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parser.add_parser('init')
    parser_1.add_argument('output_directory')
    parser_1.add_argument('--num_processes', type=int, default=1)
    parser_1.add_argument('--use_new_landmark_detector', action='store_true')
    parser_1.add_argument('--denoise_and_normalise', action='store_true')
    parser_1.add_argument('--replace_paths', type=lambda s: json.loads(s))
    parser_1.add_argument('--use_unique_parent_name', action='store_true')
    parser_1.add_argument('--speaker_embedding_path')
    parser_1.add_argument('--num_samples', type=int)
    parser_1.add_argument('--samples_path')
    parser_1.add_argument('--max_duration', type=int)
    parser_1.add_argument('--redo', action='store_true')

    parser_2 = sub_parser.add_parser('generate_file_list')

    parser_3 = sub_parser.add_parser('manifests')
    parser_3.add_argument('--dict_path')

    parser_4 = sub_parser.add_parser('vocoder')
    parser_4.add_argument('synthesis_directory')

    main(parser.parse_args())
