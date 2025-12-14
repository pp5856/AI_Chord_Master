import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import yt_dlp
import uuid

app = Flask(__name__)
CORS(app)

# 설정
UPLOAD_FOLDER = 'c:/AI_PROJECT/uploads'
MODEL_PATH = "c:/AI_PROJECT/models/chord_model.h5"
DATA_DIR = "c:/AI_PROJECT/data/processed"
INPUT_SHAPE = (84, 84, 1)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 모델 로드
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print(f"Model loaded. Classes: {len(class_names)}")

def preprocess_audio_segment(y, sr):
    """오디오 세그먼트를 CQT 이미지로 변환"""
    target_frames = INPUT_SHAPE[1]
    hop_length = 512
    
    C = librosa.cqt(y, sr=sr, n_bins=INPUT_SHAPE[0], bins_per_octave=12, hop_length=hop_length)
    C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
    
    if C_db.shape[1] > target_frames:
        C_db = C_db[:, :target_frames]
    elif C_db.shape[1] < target_frames:
        C_db = np.pad(C_db, ((0,0), (0, target_frames - C_db.shape[1])))
        
    C_db = (C_db + 80.0) / 80.0 * 255.0
    C_db = C_db[..., np.newaxis]
    C_db = C_db[np.newaxis, ...]
    return C_db

def analyze_audio_file(filepath):
    """오디오 파일 분석 및 코드 예측"""
    try:
        y, sr = librosa.load(filepath, sr=22050)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # 비트 트래킹 수행
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        results = []
        
        # 마디 단위 처리 (4비트 기준)
        beats_per_bar = 4
        
        # 비트가 너무 적으면 2초 단위로
        if len(beat_times) < 4:
            segments = np.arange(0, duration, 2.0)
            for start_time in segments:
                end_time = min(start_time + 2.0, duration)
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                y_segment = y[start_sample:end_sample]
                
                if len(y_segment) < sr * 0.5: continue
                    
                input_data = preprocess_audio_segment(y_segment, sr)
                pred = model.predict(input_data, verbose=0)
                chord = class_names[np.argmax(pred)]
                
                results.append({
                    'start': float(start_time),
                    'end': float(end_time),
                    'chord': chord
                })
        else:
            # 비트 기반 세그먼테이션
            # 첫 마디 전(Intro) 처리
            if beat_times[0] > 0.5:
                 results.append({
                    'start': 0.0,
                    'end': float(beat_times[0]),
                    'chord': 'Intro' # 혹은 첫 구간 분석
                })

            for i in range(0, len(beat_times), beats_per_bar):
                start_time = beat_times[i]
                if i + beats_per_bar < len(beat_times):
                    end_time = beat_times[i + beats_per_bar]
                else:
                    end_time = duration
                
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                y_segment = y[start_sample:end_sample]
                
                if len(y_segment) < sr * 0.5: continue

                input_data = preprocess_audio_segment(y_segment, sr)
                pred = model.predict(input_data, verbose=0)
                chord = class_names[np.argmax(pred)]
                
                results.append({
                    'start': float(start_time),
                    'end': float(end_time),
                    'chord': chord
                })
                
        return {'success': True, 'tempo': float(tempo), 'results': results}
        
    except Exception as e:
        return {'error': str(e)}

@app.route('/analyze/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        result = analyze_audio_file(filepath)
        return jsonify(result)

@app.route('/analyze/youtube', methods=['POST'])
def process_youtube():
    data = request.get_json()
    url = data.get('url')
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
        
    try:
        # yt-dlp 설정
        filename = f"{uuid.uuid4()}"
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(UPLOAD_FOLDER, filename),
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
        filepath = os.path.join(UPLOAD_FOLDER, filename + ".mp3")
        result = analyze_audio_file(filepath)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
