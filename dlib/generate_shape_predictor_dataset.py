import argparse
import pickle
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def detect_dlib_faces(args):
    from face_landmarks_server import FaceDetector, get_video_frames

    output_directory = Path(args.output_directory)
    if output_directory.exists() and args.redo:
        shutil.rmtree(output_directory)
    output_directory.mkdir(exist_ok=True)

    video_paths = list(Path(args.dataset_directory).rglob('*.mp4'))
    for video_path in tqdm(video_paths):
        video_frames = list(get_video_frames(video_path=str(video_path)))
        middle_frame_index = (len(video_frames) // 2) - 1
        middle_frame = video_frames[middle_frame_index]

        face = FaceDetector().detect(frame_index=0, frame=middle_frame, upsample_num_times=0)
        if face is None:
            continue
        face = [face.left(), face.top(), face.right(), face.bottom()]

        _id = str(uuid.uuid4())
        with output_directory.joinpath(f'{_id}.pkl').open('wb') as f:
            pickle.dump(face, f)
        cv2.imwrite(str(output_directory.joinpath(f'{_id}.png')), middle_frame)


def detect_ibug_landmarks(args):
    from helpers import init_ibug_facial_detectors, get_ibug_landmarks

    init_ibug_facial_detectors()

    frame_paths = list(Path(args.dataset_directory).glob('*.png'))

    def dataset_generator():
        for frame_path in tqdm(frame_paths):

            # load frame and face points
            frame = cv2.imread(str(frame_path))
            with frame_path.with_suffix('.pkl').open('rb') as f:
                face = pickle.load(f)

            frame_landmarks = get_ibug_landmarks(
                video_frames=[frame],
                faces=np.asarray([face]),
                _filter=False
            )[1]['landmarks'][0][0]

            # draw points of interest in debug mode
            if args.debug:
                for x, y in frame_landmarks[27:]:
                    frame = cv2.circle(frame, (int(x), int(y)), radius=0, color=(0, 0, 255), thickness=2)
                cv2.imshow('Frame', frame)
                cv2.waitKey(500)

            # only need inner face landmarks i.e. eyes, nose and mouth
            yield str(frame_path.resolve()), face, frame_landmarks[27:]

    if args.debug:
        for _ in dataset_generator():
            pass
        
        return

    # generate xml for training DLIB shape predictor
    with open(args.xml_output_path, 'w') as f:
        f.write('<dataset><images>\n')
        for frame_path, (left, top, right, bottom), frame_landmarks in dataset_generator():
            width = right - left
            height = bottom - top
            box = f'<box top=\'{top}\' left=\'{left}\' width=\'{width}\' height=\'{height}\'>'
            landmark_index = 27
            for x, y in frame_landmarks:
                box += f'<part name=\'{landmark_index}\' x=\'{int(x)}\' y=\'{int(y)}\'/>'
                landmark_index += 1
            box += '</box>'
            f.write(f'<image file=\'{frame_path}\'>{box}</image>\n')
        f.write('</images></dataset>')

    
def main(args):
    # generate dataset to train DLIB shape predictor based on LRS3 key points from the iBUG landmark predictor
    # only need inner points from face i.e. eyes, nose and mouth - makes predictor faster
    # need to use DLIB face detector though
    # use midpoint frame from each LRS3 video = ~150k images
    
    f = {
        'detect_dlib_faces': detect_dlib_faces,
        'detect_ibug_landmarks': detect_ibug_landmarks
    }
    f[args.run_type](args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parser.add_parser('detect_dlib_faces')
    parser_1.add_argument('dataset_directory')
    parser_1.add_argument('output_directory')
    parser_1.add_argument('--redo', action='store_true')

    parser_2 = sub_parser.add_parser('detect_ibug_landmarks')
    parser_2.add_argument('dataset_directory')
    parser_2.add_argument('xml_output_path')
    parser_2.add_argument('--debug', action='store_true')

    main(parser.parse_args())
