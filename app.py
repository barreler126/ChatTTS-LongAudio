import os
import re
import sys
import io
import json
import wave
from pathlib import Path
print('Starting...')
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import soundfile as sf
import ChatTTS
import datetime
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify, send_from_directory, send_file, Response, stream_with_context
import logging
from logging.handlers import RotatingFileHandler
from waitress import serve
load_dotenv()

from random import random
from modelscope import snapshot_download
import numpy as np
import time
import threading
from uilib.cfg import WEB_ADDRESS, SPEAKER_DIR, LOGS_DIR, WAVS_DIR, MODEL_DIR, ROOT_DIR
from uilib import utils, VERSION
from ChatTTS.utils.gpu_utils import select_device

from uilib.utils import is_chinese_os, modelscope_status
env_lang = os.getenv('lang', '')
if env_lang == 'zh':
    is_cn = True
elif env_lang == 'en':
    is_cn = False
else:
    is_cn = is_chinese_os()

CHATTTS_DIR = MODEL_DIR + '/pzc163/chatTTS'
_cache_dir = MODEL_DIR + '/models/pzc163--chatTTS/snapshots/master'
if os.path.exists(_cache_dir + "/config/path.yaml"):
    CHATTTS_DIR = _cache_dir
elif os.path.exists(CHATTTS_DIR + "/config/path.yaml"):
    pass
elif os.path.exists(MODEL_DIR + '/models--2Noise--ChatTTS'):
    CHATTTS_DIR = MODEL_DIR + '/models--2Noise--ChatTTS'
else:
    if modelscope_status():
        print('modelscope ok')
        CHATTTS_DIR = snapshot_download('pzc163/chatTTS', cache_dir=MODEL_DIR)
    else:
        print('from huggingface')
        CHATTTS_DIR = MODEL_DIR + '/models--2Noise--ChatTTS'
        import huggingface_hub
        os.environ['HF_HUB_CACHE'] = MODEL_DIR
        os.environ['HF_ASSETS_CACHE'] = MODEL_DIR
        huggingface_hub.snapshot_download(cache_dir=MODEL_DIR, repo_id="2Noise/ChatTTS", allow_patterns=["*.pt", "*.yaml"])

chat = ChatTTS.Chat()

# DEVICE handling: default to cpu for Spaces. Allow overriding with env var DEVICE.
# e.g. export DEVICE=cpu
device_env = os.getenv('DEVICE', os.getenv('device', 'cpu')).lower()
if device_env in ('default', 'auto', ''):
    device_torch = None
else:
    # e.g. 'cpu' or 'cuda:0' (if you really have gpu)
    try:
        device_torch = torch.device(device_env)
    except Exception:
        device_torch = torch.device('cpu')

# Pass device param to chat.load_models; if device_torch is None, leave None.
chat.load_models(
    source="custom",
    custom_path=CHATTTS_DIR,
    device=None if device_torch is None else device_torch,
    compile=True if os.getenv('compile', 'true').lower() != 'false' else False
)

# --- logging & flask init (unchanged) ---
log = logging.getLogger('werkzeug')
log.handlers[:] = []
log.setLevel(logging.WARNING)

app = Flask(
    __name__,
    static_folder=ROOT_DIR + '/static',
    static_url_path='/static',
    template_folder=ROOT_DIR + '/templates'
)

root_log = logging.getLogger()
root_log.handlers = []
root_log.setLevel(logging.WARNING)
app.logger.setLevel(logging.WARNING)
file_handler = RotatingFileHandler(LOGS_DIR + f'/{datetime.datetime.now().strftime("%Y%m%d")}.log',
                                   maxBytes=1024 * 1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(formatter)
app.logger.addHandler(file_handler)
app.jinja_env.globals.update(enumerate=enumerate)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.config.get('STATIC_FOLDER', ROOT_DIR + '/static'), filename)


@app.route('/')
def index():
    speakers = utils.get_speakers()
    return render_template(
        f"index{'' if is_cn else 'en'}.html",
        weburl=WEB_ADDRESS,
        speakers=speakers,
        version=VERSION
    )


@app.route('/config', methods=['GET'])
def get_config():
    default_voice = os.getenv('DEFAULT_VOICE', '2222').strip() or '2222'
    return jsonify({
        "code": 0,
        "default_voice": default_voice,
        "web_address": WEB_ADDRESS,
        "version": VERSION
    })


audio_queue = []


@app.route('/tts', methods=['GET', 'POST'])
def tts():
    global audio_queue
    text = request.args.get("text", "").strip() or request.form.get("text", "").strip()
    prompt = request.args.get("prompt", "").strip() or request.form.get("prompt", '')

    defaults = {
        "custom_voice": 0,
        "voice": "2222",
        "temperature": 0.3,
        "top_p": 0.7,
        "top_k": 20,
        "skip_refine": 0,
        "speed": 5,
        "text_seed": 42,
        "refine_max_new_token": 384,
        "infer_max_new_token": 2048,
        "wav": 0,
        "is_stream": 0
    }

    custom_voice = utils.get_parameter(request, "custom_voice", defaults["custom_voice"], int)
    voice = str(custom_voice) if custom_voice > 0 else utils.get_parameter(request, "voice", defaults["voice"], str)
    temperature = utils.get_parameter(request, "temperature", defaults["temperature"], float)
    top_p = utils.get_parameter(request, "top_p", defaults["top_p"], float)
    top_k = utils.get_parameter(request, "top_k", defaults["top_k"], int)
    skip_refine = utils.get_parameter(request, "skip_refine", defaults["skip_refine"], int)
    is_stream = utils.get_parameter(request, "is_stream", defaults["is_stream"], int)
    speed = utils.get_parameter(request, "speed", defaults["speed"], int)
    text_seed = utils.get_parameter(request, "text_seed", defaults["text_seed"], int)
    refine_max_new_token = utils.get_parameter(request, "refine_max_new_token", defaults["refine_max_new_token"], int)
    infer_max_new_token = utils.get_parameter(request, "infer_max_new_token", defaults["infer_max_new_token"], int)
    wav = utils.get_parameter(request, "wav", defaults["wav"], int)
    apply_term_rules = utils.get_parameter(request, "apply_term_rules", 1, int)
    apply_punct_rules = utils.get_parameter(request, "apply_punct_rules", 1, int)

    app.logger.info(f"[tts]{text=}\n{voice=},{skip_refine=}\n")
    if not text:
        return jsonify({"code": 1, "msg": "text params lost"})
    rand_spk = None
    seed_path = f'{SPEAKER_DIR}/{voice}'
    print(f'{voice=}')
    if voice.endswith('.csv') and os.path.exists(seed_path):
        rand_spk = utils.load_speaker(voice)
        print(f'当前使用音色 {seed_path=}')
    elif voice.endswith('.pt') and os.path.exists(seed_path):
        # map_location should be CPU when running on CPU-only
        map_loc = device_torch if device_torch is not None else torch.device('cpu')
        rand_spk = torch.load(seed_path, map_location=map_loc)
        print(f'当前使用音色 {seed_path=}')
    elif os.path.exists(f'{SPEAKER_DIR}/{voice}.csv'):
        rand_spk = utils.load_speaker(voice)
        print(f'当前使用音色 {SPEAKER_DIR}/{voice}.csv')

    if rand_spk is None:
        print(f'当前使用音色：根据seed={voice}获取随机音色')
        voice = int(voice) if re.match(r'^\d+$', voice) else 2222
        torch.manual_seed(voice)
        std, mean = torch.load(f'{CHATTTS_DIR}/asset/spk_stat.pt', map_location=torch.device('cpu')).chunk(2)
        rand_spk = torch.randn(768) * std + mean
        utils.save_speaker(voice, rand_spk)

    audio_files = []

    start_time = time.time()

    text_list = [t.strip() for t in text.split("\n") if t.strip()]
    new_text = utils.split_text(text_list, apply_term_rules=bool(apply_term_rules), apply_punct_rules=bool(apply_punct_rules))
    if text_seed > 0:
        torch.manual_seed(text_seed)

    params_infer_code = {'spk_emb': rand_spk, 'prompt': prompt}
    wavs = chat.infer(
        new_text,
        use_decoder=True,
        stream=True if is_stream == 1 else False,
        skip_refine_text=True if skip_refine == 1 else False,
        params_infer_code=params_infer_code
    )
    combined_wavdata = None

    end_time = time.time()
    inference_time = end_time - start_time
    inference_time_rounded = round(inference_time, 2)
    print(f"推理时长: {inference_time_rounded} 秒")

    combined_wavdata = np.array([], dtype=wavs[0][0].dtype)

    for wavdata in wavs:
        combined_wavdata = np.concatenate((combined_wavdata, wavdata[0]))

    sample_rate = 24000
    audio_duration = len(combined_wavdata) / sample_rate
    audio_duration_rounded = round(audio_duration, 2)
    print(f"音频时长: {audio_duration_rounded} 秒")

    # filename - 保持原有命名风格（确保没有截断）
    filename = datetime.datetime.now().strftime('%H%M%S_') + f"use{inference_time_rounded}s-audio{audio_duration_rounded}s-seed{voice}-te{temperature}-tp{top_p}-tk{top_k}-textlen{len(text)}.wav"
    try:
        os.makedirs(WAVS_DIR, exist_ok=True)
        sf.write(WAVS_DIR + '/' + filename, combined_wavdata, 24000)
    except Exception as e:
        app.logger.error(f"[tts] write wav failed: {e}")
        return jsonify({"code": 1, "msg": f"保存音频失败: {e}"})

    audio_files.append({
        "filename": WAVS_DIR + '/' + filename,
        "url": f"http://{request.host}/static/wavs/{filename}",
        "inference_time": inference_time_rounded,
        "audio_duration": audio_duration_rounded
    })
    result_dict = {"code": 0, "msg": "ok", "audio_files": audio_files}
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if len(audio_files) == 1:
        result_dict["filename"] = audio_files[0]['filename']
        result_dict["url"] = audio_files[0]['url']

    if wav > 0:
        return send_file(audio_files[0]['filename'], mimetype='audio/x-wav')
    else:
        return jsonify(result_dict)


@app.route('/clear_wavs', methods=['POST'])
def clear_wavs():
    dir_path = 'static/wavs'
    success, message = utils.ClearWav(dir_path)
    if success:
        return jsonify({"code": 0, "msg": message})
    else:
        return jsonify({"code": 1, "msg": message})


# Start server: use HOST and PORT env vars. Avoid auto-open by default.
try:
    port = int(os.environ.get('PORT', os.environ.get('HF_SPACE_PORT', '7860')))
    host = os.environ.get('HOST', '0.0.0.0')
    open_browser = os.environ.get('OPEN_BROWSER', 'false').lower() == 'true'
    if open_browser:
        threading.Thread(target=utils.openweb, args=(f'http://{host}:{port}',)).start()
    print(f'Starting server on {host}:{port}')
    serve(app, host=host, port=port)
except Exception as e:
    print(e)
