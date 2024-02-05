import argparse
import multiprocessing
import shutil
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytube import YouTube, exceptions as pytube_exceptions
from tqdm import tqdm

import config
from helpers import WhisperASR, convert_fps, crop_video_fast, extract_audio, get_fps, get_video_frames, init_ibug_facial_detectors, is_valid_file
from helpers import overlay_audio, save_video

TMP_VIDEO_PATH = '/tmp/video.mp4'
EXTRACTED_AUDIO_PATH = '/tmp/extracted_audio.wav'


class VideoNonEnglish(Exception):
    pass


def download_video(video_id, download_path, check_captions_language=False, max_retries=0):
    # progressive = video and audio in single file
    # grab best resolution available
    yt = YouTube(f'https://youtube.com/watch?v={video_id}')
    stream = yt.streams\
        .filter(progressive=True, file_extension='mp4')\
        .order_by('resolution')\
        .desc()\
        .first()  # required for getting captions
    
    if check_captions_language and not any([code in yt.captions for code in ['a.en', 'en']]):
        raise VideoNonEnglish()
    
    stream.download(output_path='/tmp', filename=download_path.name, max_retries=max_retries)


def download_process(process_index, clips_df, unique_video_ids, check_captions_language, run_asr, detect_language, output_directory):
    video_download_path = Path(f'/tmp/youtube_video_{process_index}.mp4')
    extracted_audio_path = Path(f'/tmp/extracted_audio_{process_index}.wav')

    already_processed_path = output_directory.joinpath('processed.txt')
    groundtruth_path = output_directory.joinpath('groundtruth.csv')
    
    asr = None
    if run_asr or detect_language:
        asr = WhisperASR(model='medium', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    failed_counter = 0

    for video_id in tqdm(unique_video_ids):
        with already_processed_path.open('a') as f:
            f.write(f'{video_id}\n')

        # download Youtube video once
        try:
            download_video(
                video_id=video_id,
                download_path=video_download_path,
                check_captions_language=check_captions_language,
                max_retries=5
            )
            failed_counter = 0
        except pytube_exceptions.VideoUnavailable:
            print(f'Process {process_index} - {video_id}: unavailable...')
            continue
        except VideoNonEnglish:
            print(f'Process {process_index} - {video_id}: non-english captions...')
            continue
        except Exception as e:
            print(f'Process {process_index} - {video_id}: failed "{traceback.format_exc()}"...')
            failed_counter += 1
            if failed_counter == 5:
                print(f'Process {process_index} fail limit reached...quitting')
                break
            continue

        # 1 video can have many clips - no point downloading the video multiple times
        for index, (_, row) in enumerate(clips_df[clips_df['Video ID'] == video_id].iterrows()):
            video_clip_path = output_directory.joinpath(f'{video_id}_{index+1}.mp4')
            
            start_time, end_time = float(row['Start Time']), float(row['End Time'])

            # crop the clips from the video
            # video_path, crop_start_time, crop_end_time, output_video_path
            crop_video_fast(
                video_path=f'"{video_download_path}"', 
                crop_start_time=start_time, 
                crop_end_time=end_time, 
                output_video_path=f'"{str(video_clip_path)}"'
            )

            if not is_valid_file(str(video_clip_path), select_stream='video'):
                video_clip_path.unlink()  # delete
                continue

            if run_asr or detect_language:
                extract_audio(str(video_clip_path), extracted_audio_path)

            # detect if first cropped clip is english language instead of whole video - faster
            if detect_language and index == 0:
                language, confidence = asr.detect_language(audio_path=extracted_audio_path)
                if language != 'en' or (language == 'en' and confidence < 0.9):
                    print(f'Process {process_index} - {video_id}: non-english...')
                    video_clip_path.unlink()
                    break

            if run_asr:
                asr_preds = asr.run(audio_path=extracted_audio_path)
                if len(asr_preds) == 0:
                    video_clip_path.unlink()
                    continue

                with groundtruth_path.open('a') as f:
                    f.write(f'{video_clip_path.stem},{asr_preds[0]}\n')


def download(args):
    output_directory = Path(args.output_directory)
    if output_directory.exists() and args.redo:
        shutil.rmtree(output_directory)
    output_directory.mkdir(exist_ok=True)

    clips_df = pd.read_csv(args.csv_path, names=['Video ID', 'Start Time', 'End Time', 'Centre X', 'Centre Y'])
    unique_video_ids = clips_df['Video ID'].unique()

    # for ignoring videos already processed
    already_processed_path = output_directory.joinpath('processed.txt')
    already_processed = set()
    if already_processed_path.exists():
        with already_processed_path.open('r') as f:
            already_processed = set(f.read().splitlines())

    to_be_processed = list(set(unique_video_ids) - already_processed)

    # divide video ids between processes - skipping those already processed
    tasks = []
    num_per_process = len(to_be_processed) // args.num_processes
    for i in range(args.num_processes):
        start_index = i * num_per_process
        end_index = start_index + num_per_process if i < args.num_processes - 1 else len(to_be_processed)

        tasks.append([
            i + 1,
            clips_df,
            to_be_processed[start_index:end_index],
            args.check_captions_language,
            args.asr,
            args.detect_language,
            output_directory
        ])

    with multiprocessing.Pool(processes=args.num_processes) as p:
        p.starmap(download_process, tasks)


def process(args):
    # TODO: face tracking?

    clips_df = pd.read_csv(args.csv_path, names=['Video ID', 'Start Time', 'End Time', 'Centre X', 'Centre Y'])
    dataset_directory = Path(args.dataset_directory)

    output_directory = dataset_directory.joinpath('processed')
    if output_directory.exists() and args.redo:
        shutil.rmtree(output_directory)
    output_directory.mkdir(exist_ok=True)

    init_ibug_facial_detectors(init_landmark_predictor=False)
    from sv2s.detector import face_detector as ibug_face_detector

    for video_clip_path in tqdm(list(dataset_directory.glob('*.mp4'))):
        processed_video_clip_path = output_directory.joinpath(video_clip_path.name)

        name_split = video_clip_path.stem.split('_')
        video_id, clip_id = name_split[0], int(name_split[-1]) - 1

        fps = get_fps(video_path=str(video_clip_path))
        if fps < (config.FPS - 5):
            print(f'{video_clip_path.stem}: low fps ({fps})...')
            continue

        if fps != config.FPS:
            convert_fps(str(video_clip_path), config.FPS, str(processed_video_clip_path))
        else:
            shutil.copyfile(video_clip_path, processed_video_clip_path)

        # iloc works for this sub-df even if rows have their own index
        row = clips_df[clips_df['Video ID'] == video_id].iloc[clip_id]
        centre_x, centre_y = row['Centre X'], row['Centre Y']  # normalised to shape of the frame
        
        video_frame_generator = get_video_frames(video_path=str(processed_video_clip_path))
        first_frame = next(video_frame_generator)
        height, width = first_frame.shape[:2]
        centre_x *= width
        centre_y *= height

        # get face bb where centre point is within bounds
        face_bb = None
        faces = ibug_face_detector(first_frame, rgb=False).tolist()
        for face in faces:
            left, top, right, bottom = face[:4]
            if left <= centre_x <= right and top <= centre_y <= bottom:
                face_bb = face[:4]
                break
        if face_bb is None:
            print(f'{video_clip_path.stem}: cannot find face bb...')
            processed_video_clip_path.unlink()
            continue
        
        # add padding to detection e.g. double face detection
        left, top, right, bottom = face_bb
        half_width = (right - left) / 2
        half_height = (bottom - top) / 2
        left -= half_width
        top -= half_height
        right += half_width
        bottom += half_height

        # cap bb to frame boundaries
        if left < 0:
            left = 0
        if top < 0:
            top = 0
        if right > width - 1:
            right = width - 1
        if bottom > height - 1:
            bottom = height - 1

        # crop frames to new bb
        frames = [first_frame] + [frame for frame in video_frame_generator]  # N, H, W, 3
        frames = np.asarray(frames)[:, int(top):int(bottom), int(left):int(right), :]
        save_video(frames, TMP_VIDEO_PATH, config.FPS, colour=True)

        # overlay audio
        extract_audio(video_path=str(processed_video_clip_path), audio_path=EXTRACTED_AUDIO_PATH)
        overlay_audio(TMP_VIDEO_PATH, EXTRACTED_AUDIO_PATH, str(processed_video_clip_path))


def main(args):
    f = {
        'download': download,
        'process': process
    }
    f[args.run_type](args)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parser.add_parser('download')
    parser_1.add_argument('csv_path')
    parser_1.add_argument('output_directory')
    parser_1.add_argument('num_processes', type=int)
    parser_1.add_argument('--check_captions_language', action='store_true')
    parser_1.add_argument('--asr', action='store_true')
    parser_1.add_argument('--detect_language', action='store_true')
    parser_1.add_argument('--redo', action='store_true')

    parser_2 = sub_parser.add_parser('process')
    parser_2.add_argument('csv_path')
    parser_2.add_argument('dataset_directory')
    parser_2.add_argument('--redo', action='store_true')

    main(parser.parse_args())
