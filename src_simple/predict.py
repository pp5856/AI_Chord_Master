import os
import numpy as np
import librosa
import tensorflow as tf
import random

# 설정
MODEL_PATH = "c:/AI_PROJECT/models/chord_model.h5"
DATA_DIR = "c:/AI_PROJECT/data/processed"
INPUT_SHAPE = (84, 84, 1)

def get_class_names():
    return sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

def preprocess_audio(file_path):
    """오디오 -> CQT 이미지 전처리"""
    try:
        y, sr = librosa.load(file_path, sr=22050)
        
        target_frames = INPUT_SHAPE[1]
        hop_length = 512
        required_samples = (target_frames - 1) * hop_length
        
        if len(y) < required_samples:
            y = np.pad(y, (0, required_samples - len(y)))
        else:
            y = y[:required_samples + hop_length]

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
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    classes = get_class_names()
    print(f"클래스: {len(classes)}개")
    
    print("모델 로드 중...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # 테스트 파일 선택
    true_label = random.choice(classes)
    class_dir = os.path.join(DATA_DIR, true_label)
    files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
    
    if not files: return
    test_file = random.choice(files)
    file_path = os.path.join(class_dir, test_file)
    
    print(f"\n🎵 파일: {true_label}/{test_file}")
    
    # 예측
    input_data = preprocess_audio(file_path)
    if input_data is None: return
        
    predictions = model.predict(input_data, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    predicted_label = classes[predicted_idx]
    confidence = predictions[0][predicted_idx] * 100
    
    print(f"\n🤖 예측: **{predicted_label}** ({confidence:.2f}%)")
    print(f"✅ 정답: {true_label}")
    
    if predicted_label == true_label:
        print("🎉 정답!")
    else:
        print("😭 오답")

if __name__ == "__main__":
    main()
