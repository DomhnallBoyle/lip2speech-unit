import argparse
import json
import multiprocessing
import pickle
import random
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import redis
import soundfile as sf
import torch
from tqdm import tqdm

import config

sys.path.append(str(config.FAIRSEQ_PATH))
from examples.textless_nlp.gslm.unit2speech.tacotron2.layers import TacotronSTFT
from helpers import alter_video_speed, calculate_ros, convert_fps, crop_video, extract_audio, generate_speaker_content_mapping, \
    get_fps, get_ibug_landmarks, get_landmarks as get_dlib_landmarks, get_mouth_frames, get_num_video_frames, _get_speaker_embedding, \
        get_speaker_embedding_video_path, get_updated_dims, get_video_duration, get_video_size, get_words_to_phonemes_d, init_ibug_facial_detectors, \
            num_transcript_phonemes, preprocess_audio, resize_video, save_video, split_list, synthesise


stft = None
cache = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT)

# TODO: 
# fake audio, mel-spec and speech units for testing etc - only needs mouth frames and speaker embedding
# create groundtruth csv files using WhisperASR from cropped videos

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

        num_video_frames = duration_num_frames

    return num_video_frames


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


def init_process(process_index, args, sample_paths):
    dlib_video_path = f'/tmp/video_{process_index}.mp4'
    extracted_audio_path = f'/tmp/extracted_audio_{process_index}.wav'
    cropped_video_path = f'/tmp/video_cropped_{process_index}.mp4'
    preprocessed_audio_path = f'/tmp/preprocessed_audio_{process_index}.wav'
    resized_video_path = f'/tmp/video_resized_{process_index}.mp4'
    ros_altered_video_path = f'/tmp/video_ros_{process_index}.mp4'

    if args.use_ibug_for_landmarks:
        init_ibug_facial_detectors()

    # get speaker embedding if applicable
    pre_loaded_speaker_embedding = None
    if args.speaker_embedding_path:
        pre_loaded_speaker_embedding = np.load(args.speaker_embedding_path)
    elif args.speaker_embedding_audio_path:
        pre_loaded_speaker_embedding = _get_speaker_embedding(audio_path=args.speaker_embedding_audio_path)

    words_to_phonemes_d = get_words_to_phonemes_d(language='en')

    for sample_path in tqdm(sample_paths):

        # create dataset from .npz or .mp4 files
        if sample_path.suffix == '.npz':
            # can't use original mouth frames because they are 20 FPS
            sample = np.load(sample_path, allow_pickle=True)['sample']
            video_path, speaker_embedding = sample[0], sample[2]
        else:
            # .mp4
            video_path = sample_path

            if args.speaker_content_mapping:
                speaker_embedding_video_path = get_speaker_embedding_video_path(
                    speaker_content_mapping=args.speaker_content_mapping, 
                    video_path=video_path, 
                    speaker_id_index=args.speaker_id_index, 
                )
            elif args.se_duration:
                video_duration = get_video_duration(video_path=video_path)
                start_time = np.random.uniform(0, video_duration - args.se_duration)  # includes low, exludes high
                end_time = start_time + args.se_duration
                crop_video(
                    video_path=video_path, 
                    crop_start_time=start_time, 
                    crop_end_time=end_time, 
                    output_video_path=cropped_video_path
                )
                speaker_embedding_video_path = cropped_video_path
            else:
                speaker_embedding_video_path = video_path

            extract_audio(video_path=speaker_embedding_video_path, audio_path=extracted_audio_path)
            if args.denoise_and_normalise:
                preprocess_audio(audio_path=extracted_audio_path, output_path=preprocessed_audio_path)
                extracted_audio_path = preprocessed_audio_path
            speaker_embedding = _get_speaker_embedding(audio_path=extracted_audio_path)
        
        if args.replace_paths:
            for k, v in args.replace_paths.items():
                video_path = str(video_path).replace(k, v)

        if config.DEBUG:
            print(f'Processing {video_path}...')

        video_path = Path(video_path)
        assert video_path.exists(), f'{video_path} does not exist'

        name = video_path.stem
        if args.use_unique_parent_name:
            name = f'{video_path.parents[0].name}_{name}'

        video_path = str(video_path)

        # get FPS, ignore video if too low
        fps = get_fps(video_path=video_path)
        if args.fps_buffer and fps < (config.FPS - args.fps_buffer):
            print(f'{video_path}: low fps ({fps})...')
            continue

        # resize if necessary
        if args.resize:
            width, height = get_video_size(video_path=video_path)
            video_dims = get_updated_dims(width=width, height=height)
            if video_dims != (width, height):
                resize_video(video_path, *video_dims, output_video_path=resized_video_path)
                video_path = resized_video_path

        # copy raw video or convert fps if necessary
        # ensure not making changes to original video
        raw_video_path = str(args.video_raw_directory.joinpath(f'{name}.mp4'))
        if fps != config.FPS:
            convert_fps(video_path, config.FPS, raw_video_path)
        else:
            shutil.copyfile(video_path, raw_video_path)
        video_path = raw_video_path

        # alter ROS if necessary
        if args.optimum_ros:
            response = synthesise(host='https://127.0.0.1:5002', video_path=video_path)
            asr_preds = response.json()['asrPredictions']
            if not asr_preds:
                continue
            try:
                ros = calculate_ros(
                    transcript=asr_preds[0], 
                    duration=get_video_duration(video_path=video_path),
                    ros_f=num_transcript_phonemes,
                    words_to_phonemes_d=words_to_phonemes_d
                )
            except Exception as e:
                print('Failed to calculate ROS:', e)
                continue
            ros_ratio = args.optimum_ros / ros
            if not 0.5 <= ros_ratio <= 100:
                continue
            alter_video_speed(
                video_path=video_path,
                output_video_path=ros_altered_video_path,
                speed=ros_ratio
            )
            shutil.copy(ros_altered_video_path, video_path)

        num_video_frames, video_duration = trim_video_to_duration(
            video_path=video_path, 
            cropped_video_path=cropped_video_path
        )
        if args.max_duration:
            num_video_frames = crop_to_random_duration(
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

        # save speaker embedding - pre-loaded se given priority
        speaker_embedding_path = args.spk_emb_directory.joinpath(f'{name}.npy')
        speaker_embedding = pre_loaded_speaker_embedding if pre_loaded_speaker_embedding is not None else speaker_embedding
        assert speaker_embedding.shape == (256,) and speaker_embedding.dtype == np.float32
        np.save(speaker_embedding_path, speaker_embedding)

        # use better detector if applicable
        if config.DEBUG: 
            print('Getting face landmarks...')
        face_landmarks = None
        if args.use_ibug_for_landmarks:
            face_landmarks = get_ibug_landmarks(video_path=video_path, skip_nth_frame=args.skip_nth_frame)[1]
        elif args.use_dlib_for_landmarks:
            # dlib docker image needs video copied to shared volume to access
            shutil.copyfile(video_path, dlib_video_path)
            face_landmarks = get_dlib_landmarks(redis_cache=cache, video_path=dlib_video_path, max_frames=num_video_frames)

        if face_landmarks is not None:
            # for testing purposes, just include test samples that have landmarks for every frame
            # usually, iBUG detector/predictors are very good at extreme poses so it usually always has landmarks
            num_invalid_frames = sum([l is None for l in face_landmarks])
            if num_invalid_frames > 0:
                if config.DEBUG:
                    print(f'{num_invalid_frames}/{num_video_frames} invalid frames...')
                continue

            with args.landmarks_directory.joinpath(f'{name}.pkl').open('wb') as f:
                pickle.dump(face_landmarks, f)

            if args.extract_mouth_frames:
                mouth_video_path = str(args.video_directory.joinpath(f'{name}.mp4'))
                face_landmarks = [np.asarray(l) for l in face_landmarks]
                mouth_frames = get_mouth_frames(video_path=video_path, landmarks=face_landmarks, greyscale=False)
                if mouth_frames is None:
                    continue
                save_video(mouth_frames, mouth_video_path, fps=config.FPS, colour=True)

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
    args.video_directory = video_directory
    args.mel_spec_directory = mel_spec_directory
    args.spk_emb_directory = spk_emb_directory
    args.landmarks_directory = landmarks_directory
    args.processed_path = output_directory.joinpath(f'{args.type}_processed.txt')

    if args.samples_path:
        with open(args.samples_path, 'r') as f:
            sample_paths = set(f.read().splitlines())
    else:
        sample_paths = list(dataset_directory.glob(args.glob))
        if args.num_samples:
            random.shuffle(sample_paths)
            sample_paths = sample_paths[:args.num_samples]

    # don't redo already processed samples
    if args.processed_path.exists():
        with args.processed_path.open('r') as f:
            already_processed = set(f.read().splitlines())

        sample_paths = [p for p in sample_paths if str(p) not in already_processed]

    args.speaker_content_mapping = generate_speaker_content_mapping(sample_paths, args.speaker_id_index) if args.use_se_content_mapping else None

    tasks = [[i, args, _sample_paths]
             for i, _sample_paths in enumerate(split_list(sample_paths, args.num_processes))]
    with multiprocessing.Pool(processes=args.num_processes) as p:
        p.starmap(init_process, tasks)


def generate_file_list(args):
    dataset_directory = Path(args.dataset_directory)

    names = []
    for video_path in dataset_directory.joinpath(f'video/{args.type}').glob('*.mp4'):
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
        shutil.copy(args.dict_path, dataset_directory.joinpath('label'))


def vocoder(args):
    dataset_directory = Path(args.dataset_directory)
    assert dataset_directory.exists()

    synthesis_directory = Path(args.synthesis_directory)
    assert synthesis_directory.exists()

    output_directory = synthesis_directory.joinpath('vocoder')
    if output_directory.exists() and args.redo:
        shutil.rmtree(output_directory)
    output_directory.mkdir(exist_ok=True)

    # TODO: create symlinks rather than copying files
    for source_d, dest_d in zip([
            dataset_directory.joinpath(f'audio/{args.type}'),
            dataset_directory.joinpath('label'),
            dataset_directory.joinpath(f'spk_emb/{args.type}'),
            synthesis_directory.joinpath(f'pred_mel/{args.type}')
        ], [
            output_directory.joinpath(f'audio/{args.type}'),
            output_directory.joinpath('label'),
            output_directory.joinpath(f'spk_emb/{args.type}'),
            output_directory.joinpath(f'mel/{args.type}')
        ]):
        if dest_d.exists():
            continue
        print(f'Copying {source_d} to {dest_d}...')
        shutil.copytree(source_d, dest_d)

    manifest_path = output_directory.joinpath(f'label/{args.type}.tsv')
    unt_path = output_directory.joinpath(f'label/{args.type}.unt')

    with manifest_path.open('r') as f:
        manifest = f.read().splitlines()
        manifest[0] = str(output_directory.resolve())

    text_labels_f = output_directory.joinpath(f'label/{args.type}.txt').open('w') if args.text_labels else None

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

            # combine text labels if applicable
            if text_labels_f:
                pred_t_labels_path = synthesis_directory.joinpath(f'pred_text/{_id}.txt')
                if not pred_t_labels_path.exists():
                    raise Exception(f'Missing text labels for {_id}...')
                with pred_t_labels_path.open('r') as f4:
                    t_labels = f4.read().splitlines()[0]
                text_labels_f.write(f'{t_labels}\n')

    if text_labels_f:
        text_labels_f.close()


def combine(args):
    dataset_directory = Path(args.dataset_directory)
    if dataset_directory.exists():
        shutil.rmtree(dataset_directory)
    dataset_directory.mkdir()

    combine_directories = [Path(p) for p in args.directories]
    assert all([p.exists() for p in combine_directories])

    def combine(p, _type):
        print(f'Combining {_type} in {str(p)}...')

        # create any new directories if necessary
        for sub_d in [f'{_type}', f'video/{_type}', f'audio/{_type}', f'spk_emb/{_type}', f'mel/{_type}', 'label']:
            dataset_directory.joinpath(sub_d).mkdir(exist_ok=True, parents=True)

        # gather tsv and unt lines
        with p.joinpath(f'label/{_type}.tsv').open('r') as f:
            tsv_lines = f.read().splitlines()[1:]  # ignore first line (dataset absolute path)
        
        with p.joinpath(f'label/{_type}.unt').open('r') as f:
            unt_lines = f.read().splitlines()

        assert len(tsv_lines) == len(unt_lines)

        # create symlinks, gather new tsv lines
        new_tsv_lines, new_unt_lines = [], []
        for tsv_line, unt_line in tqdm(zip(tsv_lines, unt_lines)):
            name, _, _, num_video_frames, num_audio_frames = tsv_line.split('\t')
            new_name = f'{_type}/{str(uuid.uuid4())}'

            if not p.joinpath(f'{name}.mp4').exists():
                continue

            for rel_file_path, new_rel_file_path in zip(
                [f'{name}.mp4', f'video/{name}.mp4', f'audio/{name}.wav', f'spk_emb/{name}.npy', f'mel/{name}.npy'],
                [f'{new_name}.mp4', f'video/{new_name}.mp4', f'audio/{new_name}.wav', f'spk_emb/{new_name}.npy', f'mel/{new_name}.npy']
            ):
                abs_file_path = p.joinpath(rel_file_path)
                assert abs_file_path.exists()

                new_abs_file_path = dataset_directory.joinpath(new_rel_file_path)
                new_abs_file_path.symlink_to(abs_file_path)

            new_tsv_lines.append([new_name, num_video_frames, num_audio_frames])
            new_unt_lines.append(unt_line)

        new_tsv_file_path = dataset_directory.joinpath(f'label/{_type}.tsv')

        # write header line in new tsv file
        if not new_tsv_file_path.exists():
            with new_tsv_file_path.open('w') as f:
                f.write(f'{str(dataset_directory.resolve())}\n')

        # write new tsv and unt files
        with new_tsv_file_path.open('a') as f:
            for name, num_video_frames, num_audio_frames in new_tsv_lines:
                f.write(f'{name}\tvideo/{name}.mp4\taudio/{name}.wav\t{num_video_frames}\t{num_audio_frames}\n')

        with dataset_directory.joinpath(f'label/{_type}.unt').open('a') as f:
            for unt in new_unt_lines:
                f.write(f'{unt}\n')

    # combine directories into 1, including all sub-directories
    for p in combine_directories:
        for _type in ['train', 'val', 'test']:
            if p.joinpath(_type).exists():  # ensure sub-d exists before continuing
                combine(p, _type)

    # copy the dict path for training
    shutil.copy(args.dict_path, dataset_directory.joinpath('label'))


def create(args):
    f = {
        'init': init,
        'generate_file_list': generate_file_list,
        'manifests': manifests,
        'vocoder': vocoder,
    }
    f[args.create_type](args)


def main(args):
    f = {
        'create': create,
        'combine': combine
    }
    f[args.run_type](args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_directory')

    sub_parser = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parser.add_parser('create')
    parser_1.add_argument('type', choices=['train', 'val', 'test'])

    sub_parser_1 = parser_1.add_subparsers(dest='create_type')

    parser_1a = sub_parser_1.add_parser('init')
    parser_1a.add_argument('output_directory')
    parser_1a.add_argument('--glob', default='*.npz')
    parser_1a.add_argument('--resize', action='store_true')
    parser_1a.add_argument('--num_processes', type=int, default=1)
    parser_1a.add_argument('--use_ibug_for_landmarks', action='store_true')
    parser_1a.add_argument('--skip_nth_frame', type=int, default=1)
    parser_1a.add_argument('--use_dlib_for_landmarks', action='store_true')
    parser_1a.add_argument('--extract_mouth_frames', action='store_true')
    parser_1a.add_argument('--denoise_and_normalise', action='store_true')
    parser_1a.add_argument('--replace_paths', type=lambda s: json.loads(s))
    parser_1a.add_argument('--use_unique_parent_name', action='store_true')
    parser_1a.add_argument('--use_se_content_mapping', action='store_true')
    parser_1a.add_argument('--se_duration', type=float)  # randomly crop video to this duration for se
    parser_1a.add_argument('--speaker_id_index', type=int, default=-2)
    parser_1a.add_argument('--speaker_embedding_path')
    parser_1a.add_argument('--speaker_embedding_audio_path')
    parser_1a.add_argument('--num_samples', type=int)
    parser_1a.add_argument('--samples_path')
    parser_1a.add_argument('--max_duration', type=int)
    parser_1a.add_argument('--fps_buffer', type=int)
    parser_1a.add_argument('--optimum_ros', type=float)
    parser_1a.add_argument('--redo', action='store_true')

    parser_1b = sub_parser_1.add_parser('generate_file_list')

    parser_1c = sub_parser_1.add_parser('manifests')
    parser_1c.add_argument('--dict_path')

    parser_1d = sub_parser_1.add_parser('vocoder')
    parser_1d.add_argument('synthesis_directory')
    parser_1d.add_argument('--text_labels', action='store_true')
    parser_1d.add_argument('--redo', action='store_true')

    parser_2 = sub_parser.add_parser('combine')
    parser_2.add_argument('directories', type=lambda s: s.split(','))
    parser_2.add_argument('dict_path')

    main(parser.parse_args())
