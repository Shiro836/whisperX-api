from flask import Flask, request
import time
import base64
import io
from threading import Lock
import whisperx
import torch

batch_size = 16
device = 'cuda'
compute_type = 'float16'

app = Flask(__name__)


@app.route('/ping')
async def ping():
    return 'pong'


whisper_model = None

def load_model():
    global whisper_model
    whisper_model = whisperx.load_model("large-v3", device, compute_type='float16')

transcribe_mutex = Lock()

@app.route('/transcribe', methods=['POST'])
async def transcribe():
    with transcribe_mutex:
        start = time.time()

        audio_input_data = request.json['audio']
        audio_input_data = base64.b64decode(audio_input_data)

        if whisper_model is None:
            load_model()

        audio = whisperx.load_audio(audio_input_data)

        with torch.no_grad():
            result = whisper_model.transcribe(audio, batch_size=batch_size)
        print(result)

        print(f"Time taken: {time.time() - start}")

        torch.cuda.empty_cache()
        
        return result["segments"]


align_model = None
align_metadata = None

def load_align_model():
    global align_model
    global align_metadata
    align_model, align_metadata = whisperx.load_align_model(model_name='WAV2VEC2_ASR_LARGE_LV60K_960H', language_code='en', device=device)

align_mutex = Lock()

@app.route('/align', methods=['POST'])
async def align():
    with align_mutex:
        start = time.time()

        audio_input_data = request.json['audio']
        audio_input_data = base64.b64decode(audio_input_data)
        transcript = request.json['transcript']

        if align_model is None:
            load_align_model()

        audio = whisperx.load_audio(audio_input_data)

        with torch.no_grad():
            result = whisperx.align(transcript, align_model, align_metadata, audio, device, return_char_alignments=False)
        print(result)

        print(f"Time taken: {time.time() - start}")

        torch.cuda.empty_cache()

        return result


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8777, debug=True)
