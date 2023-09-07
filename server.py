import argparse
import shutil
import subprocess
import sys
import time
import threading
import uuid
from http import HTTPStatus
from os.path import exists
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import requests
import soundfile as sf
import torch
from flask import Flask, current_app, g as app_context, redirect, request, url_for

from create_dataset import manifests as create_manifests, vocoder as setup_vocoder_inference
sys.path.append('/home/domhnall/Repos/sv2s')
from asr import WhisperASR
from utils import convert_fps, convert_video_codecs, get_fps, get_speaker_embedding, get_video_frames, overlay_audio

FPS = 25
SAMPLING_RATE = 16000
AUDIO_FRAME_RATE = 50
STATIC_PATH = Path('static')
SERVER_PATH = Path('/tmp/server')
INPUTS_PATH = SERVER_PATH.joinpath('inputs')
WORKING_DIRECTORY = Path('/tmp/lip2speech')
TYPE = 'test'
VIDEO_RAW_DIRECTORY = WORKING_DIRECTORY.joinpath(f'{TYPE}')
AUDIO_DIRECTORY = WORKING_DIRECTORY.joinpath(f'audio/{TYPE}')
VIDEO_DIRECTORY = WORKING_DIRECTORY.joinpath(f'video/{TYPE}')
MEL_SPEC_DIRECTORY = WORKING_DIRECTORY.joinpath(f'mel/{TYPE}')
SPK_EMB_DIRECTORY = WORKING_DIRECTORY.joinpath(f'spk_emb/{TYPE}')
LANDMARKS_DIRECTORY = WORKING_DIRECTORY.joinpath(f'landmark/{TYPE}')
LABEL_DIRECTORY = WORKING_DIRECTORY.joinpath('label')
SYNTHESIS_DIRECTORY = WORKING_DIRECTORY.joinpath('synthesis_results')
VOCODER_DIRECTORY = WORKING_DIRECTORY.joinpath('vocoder_results')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using:', device)
asr = WhisperASR(model='tiny.en', device=device)
sem = threading.Semaphore()


def run_command(s):
    subprocess.run(s, shell=True)


def web_app():
    setup()

    app = Flask(__name__, static_folder=str(STATIC_PATH))
    app.secret_key = str(uuid.uuid4())

    @app.before_request
    def incoming_request():
        app_context.start_time = time.time()

    @app.after_request
    def outgoing_response(response):
        time_taken = round(time.time() - app_context.start_time, 2)
        current_app.logger.info('%s secs %s %s %s', time_taken, request.method, request.path, dict(request.args))

        return response

    @app.get('/')
    def index():
        return redirect(url_for('demo'))

    @app.get('/demo')
    def demo():
        html = """
            <!DOCTYPE html>
                <head>
                    <title>VSG Demo</title>
                </head>
                <body>
                    <h2><u>Visual Speech Synthesis:</u></h2>

                    <div>
                        <div id="controls">

                            <div id="audio-selection">
                                <h3>Audio Upload</h3>
                                <input id="audio-upload" type="file"/><br>
                            </div>

                            <br><hr>

                            <div id="recorded-input">
                                <h3>Webcam</h3>
                                <video id="video" height="400" width="600" autoplay></video><br><br>
                                <button id="start">Record</button>
                                <button id="stop">Synthesise</button><br>
                            </div>

                            <br><hr>

                            <div id="uploaded-input">
                                <h3>Video Upload</h3>
                                <input id="video-upload" type="file"/><br>
                                <button id="upload">Synthesise</button><br>
                            </div>

                        </div>

                        <div>
                            <h3>History</h3>
                            <button id="clear">Clear All</button>
                            <div id="history"></div>
                        </div>
                    </div>
                </body>

                <style>
                    html, body {
                        margin: 0; 
                        height: 100%; 
                        overflow: hidden
                    }
                    #controls {
                        width: 50%;
                        float: left;
                        overflow: auto;
                        max-height: 80vh;
                    }
                    #history {
                        width: 50%;
                        float: right;
                        overflow: auto;
                        max-height: 80vh;
                    }
                    .history-element {
                        height: 425px;
                    }
                    .video-div {
                        width: 60%;
                        float: left;
                        height: 100%;
                    }
                    .stats-div {
                        width: 40%;
                        float: right;
                        height: 100%;
                    }
                    #controls hr {
                        width: 75%;
                        margin-left: 0;
                    }
                    fieldset {
                        min-width: auto;
                        display: inline;
                    }
                    .history-element .history-counter {
                        display: inline;
                    }
                </style>

                <script>
                    var video, startBtn, stopBtn, uploadBtn, clearBtn, stream, recorder, globalVideoData, historyCounter = 0;

                    video = document.getElementById("video");
                    startBtn = document.getElementById("start");
                    stopBtn = document.getElementById("stop");
                    uploadBtn = document.getElementById("upload");
                    clearBtn = document.getElementById("clear");

                    startBtn.onclick = startRecording;
                    stopBtn.onclick = stopRecording;
                    uploadBtn.onclick = uploadVideo;
                    clearBtn.onclick = clearHistory;
                    startBtn.disabled = true;
                    stopBtn.disabled = true;

                    navigator.mediaDevices.getUserMedia({
                        video: {
                            frameRate: {
                                ideal: 20,
                                min: 20,
                                max: 30
                            }
                        },
                        audio: true
                    })
                    .then(stm => {
                        stream = stm;
                        startBtn.removeAttribute("disabled");
                        video.srcObject = stream;
                    }).catch(e => console.error(e));

                    function synthesise(videoData) {
                        var audioFile = document.getElementById("audio-upload").files[0];
                        if (audioFile == null) {
                            alert("Please select an audio file");
                            stopBtn.textContent = "Synthesise";
                            uploadBtn.textContent = "Synthesise";
                            return;
                        }

                        // create form data
                        var formData = new FormData();
                        formData.append("video", videoData);
                        formData.append("audio", audioFile);

                        var xhr = new XMLHttpRequest();
                        xhr.addEventListener("load", function(event) {
                            var jsonResponse = xhr.response;
                            if (xhr.status != 200) {
                                alert(jsonResponse["message"]);
                            } else {
                                var jsonResponse = xhr.response;
                                var videoURL = jsonResponse["video_url"];
                                var asrPredictions = jsonResponse["asr_predictions"];

                                var asrText = "Whisper ASR Predictions:<br>";                      
                                if (asrPredictions.length > 0) {
                                    for (i = 0; i < asrPredictions.length; i++) {
                                        asrText += (i+1) + ": " + asrPredictions[i] + "<br>";
                                    }
                                } else {
                                    asrText += "None";
                                }

                                appendToHistory(videoURL, asrText);
                            }
                            startBtn.removeAttribute("disabled");
                            stopBtn.textContent = "Synthesise";
                            uploadBtn.removeAttribute("disabled");
                            uploadBtn.textContent = "Synthesise";
                        });
                        xhr.open("POST", "/synthesise");
                        xhr.responseType = "json";
                        xhr.send(formData);
                    }

                    function startRecording() {
                        recorder = new MediaRecorder(stream, {
                            mimeType: "video/webm"
                        });
                        recorder.start();
                        stopBtn.removeAttribute("disabled");vocoder
                        startBtn.disabled = true;
                        startBtn.textContent = "Recording...";
                    }

                    function stopRecording() {
                        recorder.ondataavailable = e => {
                            startBtn.textContent = "Record";
                            stopBtn.disabled = true;
                            stopBtn.textContent = "Please Wait...";                   
                            synthesise(e.data);
                            globalVideoData = e.data;
                        };
                        recorder.stop();
                    }

                    function uploadVideo() {
                        var videoFile = document.getElementById("video-upload").files[0];
                        if (videoFile == null) {
                            alert("Please select a video file");
                        } else {
                            uploadBtn.disabled = true;
                            uploadBtn.textContent = "Please Wait...";
                            synthesise(videoFile);
                            globalVideoData = videoFile;
                        }
                    }

                    function appendToHistory(videoURL, asrText) {
                        var div = document.createElement("div");
                        var videoDiv = document.createElement("div");
                        var statsDiv = document.createElement("div");

                        div.className = "history-element";
                        videoDiv.className = "video-div";
                        statsDiv.className = "stats-div";

                        div.innerHTML = '<br><p class="history-counter">' + ++historyCounter + ': </p><button class="remove-element-btn" onclick="removeFromHistory(this)">Remove</button><br>';
                        videoDiv.innerHTML = '<video src="' + videoURL + '" height="400" width="600" type="video/mp4" controls></video>';
                        statsDiv.innerHTML += asrText;
                        div.appendChild(videoDiv);
                        div.appendChild(statsDiv);

                        document.getElementById("history").appendChild(div);
                        div.scrollIntoView();
                    }

                    function removeFromHistory(button) {
                        // get containing div
                        var parentDiv = button.parentNode;
                        parentDiv.remove();

                        // fix history counter
                        historyCounter = 0;
                        var historyDiv = document.getElementById("history");
                        var historyElements = historyDiv.childNodes;
                        for (var i = 0; i < historyElements.length; i++) {
                            var historyElement = historyElements[i];
                            historyElement.getElementsByClassName("history-counter")[0].innerText = ++historyCounter + ": ";
                        }
                    }

                    function clearHistory() {
                        document.getElementById("history").innerHTML = "";
                        historyCounter = 0;
                    }

                </script>
            </html>
        """

        return html

    @app.post('/synthesise')
    def synthesise():
        # acquire lock
        sem.acquire()

        video_file = request.files['video']
        audio_file = request.files['audio']

        name = str(uuid.uuid4())
        video_upload_path = str(INPUTS_PATH.joinpath(f'{name}.mp4'))
        audio_upload_path = str(INPUTS_PATH.joinpath(f'{name}.wav'))
        video_raw_path = str(VIDEO_RAW_DIRECTORY.joinpath(f'{name}.mp4'))
        audio_path = str(AUDIO_DIRECTORY.joinpath(f'{name}.wav'))
        pred_audio_path = str(VOCODER_DIRECTORY.joinpath(f'pred_wav/{TYPE}/{name}.wav'))
        video_download_path = str(STATIC_PATH.joinpath(f'{name}.mp4'))

        video_file.save(video_upload_path)
        audio_file.save(audio_upload_path)

        # setup directory
        if WORKING_DIRECTORY.exists():
            shutil.rmtree(WORKING_DIRECTORY)
        for d in [WORKING_DIRECTORY, VIDEO_RAW_DIRECTORY, AUDIO_DIRECTORY, VIDEO_DIRECTORY, MEL_SPEC_DIRECTORY, SPK_EMB_DIRECTORY, LANDMARKS_DIRECTORY, LABEL_DIRECTORY]:
            d.mkdir(parents=True)

        # convert fps if applicable
        if get_fps(video_path=video_upload_path) != FPS:
            convert_fps(input_video_path=video_upload_path, fps=FPS, output_video_path=video_raw_path)
        else:
            shutil.copyfile(video_upload_path, video_raw_path)

        num_video_frames = len(get_video_frames(video_path=video_raw_path))
        video_duration = num_video_frames / FPS
        num_audio_frames = int(video_duration * SAMPLING_RATE)

        # extract audio
        audio = np.random.rand(num_audio_frames).astype(np.float32)
        sf.write(audio_path, audio, SAMPLING_RATE)

        # extract mel spec
        mel = np.random.rand(80, 100).astype(np.float32)
        np.save(MEL_SPEC_DIRECTORY.joinpath(f'{name}.npy'), mel)

        # get speaker embedding
        try:
            speaker_embedding = get_speaker_embedding(audio_path=audio_upload_path)
        except ConnectionError:
            return {'message': 'Speaker embedding server not available'}, HTTPStatus.INTERNAL_SERVER_ERROR
        speaker_embedding = np.asarray(speaker_embedding, dtype=np.float32)
        assert speaker_embedding.shape == (256,) and speaker_embedding.dtype == np.float32
        np.save(SPK_EMB_DIRECTORY.joinpath(f'{name}.npy'), speaker_embedding)

        # create file.list for extracting mouth frames
        with open(WORKING_DIRECTORY.joinpath(f'{TYPE}_file.list'), 'w') as f:
            f.write(f'{TYPE}/{name}\n')

        # extract mouth frames
        response = requests.post('http://127.0.0.1:5001/extract_mouth_frames', json={'root': str(WORKING_DIRECTORY)})
        if response.status_code != HTTPStatus.NO_CONTENT:
            return {'message': 'Failed to extract mouth frames'}, HTTPStatus.INTERNAL_SERVER_ERROR

        # create manifests
        create_manifests(SimpleNamespace(**{'type': TYPE, 'dataset_directory': WORKING_DIRECTORY}))

        # extract speech units
        num_speech_units = int(video_duration * AUDIO_FRAME_RATE)
        speech_units = ['14'] * num_speech_units
        with LABEL_DIRECTORY.joinpath('test.unt').open('w') as f:
            f.write(f'{" ".join(speech_units)}\n')

        # run synthesis
        run_command(f'export CUDA_VISIBLE_DEVICES="{device}"; ./synthesise.sh {LABEL_DIRECTORY} {SYNTHESIS_DIRECTORY}')

        # setup vocoder directory
        setup_vocoder_inference(SimpleNamespace(**{'type': TYPE, 'dataset_directory': WORKING_DIRECTORY, 'synthesis_directory': SYNTHESIS_DIRECTORY}))

        # run vocoder
        run_command(f'export CUDA_VISIBLE_DEVICES="{device}"; ./vocoder.sh {SYNTHESIS_DIRECTORY.joinpath("vocoder")} {VOCODER_DIRECTORY}')

        # overlay onto video
        overlay_audio(video_raw_path, pred_audio_path, video_upload_path)

        # browser video playback compatibility, h264 is pretty universal
        convert_video_codecs(
            input_video_path=video_upload_path,
            video_codec='libx264',
            audio_codec='aac',
            output_video_path=video_download_path
        )

        # get asr results
        asr_preds = asr.run(pred_audio_path)

        # release lock
        sem.release()

        return {
            'video_url': url_for('static', filename=Path(video_download_path).name),
            'asr_predictions': asr_preds
        }

    return app


def setup():
    # setup static and server directories
    for d in [INPUTS_PATH, STATIC_PATH]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)


def main(args):
    setup()
    app = web_app()
    app.run('0.0.0.0', port=args.port, debug=args.debug)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5002)
    parser.add_argument('--debug', action='store_true')

    main(parser.parse_args())

