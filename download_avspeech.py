import argparse
import http
import shutil
from pathlib import Path

import pandas as pd
import torch
from pytube import YouTube, exceptions as pytube_exceptions
from tqdm import tqdm

from helpers import WhisperASR, crop_video, extract_audio, is_valid_file

VIDEO_DOWNLOAD_PATH = '/tmp/youtube_video.mp4'
EXTRACTED_AUDIO_PATH = '/tmp/extracted_audio.wav'

# TODO: 
# avspeech only has centre point of face, doesn't have bounding box
# would need to do face detection after, check if centre point in bb
# multiple faces per video, not clear if max bb would work
# can still get face centre point by name of cropped video file - pandas retains ordering when filtering
# crop out the face + padding from each video


def download_video(video_id, max_retries=0):
    YouTube(f'https://www.youtube.com/watch?v={video_id}').streams.first().download(
        output_path='/tmp',
        filename='youtube_video.mp4',
        max_retries=max_retries
    )


def main(args):
    csv_path = Path(args.csv_path)

    output_directory = Path(args.output_directory)
    if output_directory.exists() and args.redo:
        shutil.rmtree(output_directory)
    output_directory.mkdir(exist_ok=True)

    groundtruth_path = output_directory.joinpath('groundtruth.csv')

    # for ignoring videos already processed
    already_processed_path = output_directory.joinpath('processed.txt')
    already_processed = set()
    if already_processed_path.exists():
        with already_processed_path.open('r') as f:
            already_processed = set(f.read().splitlines())

    clips_df = pd.read_csv(csv_path, names=['Video ID', 'Start Time', 'End Time', 'Centre X', 'Centre Y'])
    unique_video_ids = clips_df['Video ID'].unique()

    asr = WhisperASR(model='medium', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    for video_id in tqdm(unique_video_ids):
        if video_id in already_processed:
            continue
        else:
            with already_processed_path.open('a') as f:
                f.write(f'{video_id}\n')

        # download Youtube video once
        try:
            download_video(video_id=video_id)
        except pytube_exceptions.VideoUnavailable:
            print(f'{video_id}: unavailable...')
            continue
        except Exception as e:
            print(f'{video_id}: failed "{e}" ... retrying')
            download_video(video_id=video_id, max_retries=5)

        # 1 video can have many clips - no point downloading the video multiple times
        for index, (_, row) in enumerate(clips_df[clips_df['Video ID'] == video_id].iterrows()):
            video_clip_path = output_directory.joinpath(f'{video_id}_{index+1}.mp4')
            
            start_time, end_time = float(row['Start Time']), float(row['End Time'])

            # crop the clips from the video
            crop_video(f'"{VIDEO_DOWNLOAD_PATH}"', start_time, end_time, f'"{str(video_clip_path)}"')

            if not is_valid_file(str(video_clip_path), select_stream='video'):
                video_clip_path.unlink()  # delete
                continue

            extract_audio(str(video_clip_path), EXTRACTED_AUDIO_PATH)
            
            # detect if first cropped clip is english language instead of whole video - faster
            if index == 0:
                language = asr.detect_language(audio_path=EXTRACTED_AUDIO_PATH)
                if language != 'en':
                    print(f'{video_id}: non-english...')
                    video_clip_path.unlink()
                    break

            asr_preds = asr.run(audio_path=EXTRACTED_AUDIO_PATH)
            if len(asr_preds) == 0:
                video_clip_path.unlink()
                continue

            with groundtruth_path.open('a') as f:
                f.write(f'{video_clip_path.stem},{asr_preds[0]}\n')

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path')
    parser.add_argument('output_directory')
    parser.add_argument('--redo', action='store_true')

    main(parser.parse_args())
