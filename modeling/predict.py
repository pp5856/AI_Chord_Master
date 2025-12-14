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
    """디렉토리에서 클래스 목록 로드"""
    return sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

def preprocess_audio(file_path):
    """오디오 파일을 모델 입력용 CQT 이미지로 변환"""
    try:
        # 1. 오디오 로드
        y, sr = librosa.load(file_path, sr=22050)
        
        # 2. 길이 맞추기
        target_frames = INPUT_SHAPE[1]
        hop_length = 512
        required_samples = (target_frames - 1) * hop_length
        
        if len(y) < required_samples:
            y = np.pad(y, (0, required_samples - len(y)))
        else:
            y = y[:required_samples + hop_length]

        # 3. CQT 변환
        C = librosa.cqt(y, sr=sr, 
                       n_bins=INPUT_SHAPE[0], 
                       bins_per_octave=12, 
                       hop_length=hop_length)
        
        # 4. dB 변환
        C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
        
        # 5. 크기 맞추기
        if C_db.shape[1] > target_frames:
            C_db = C_db[:, :target_frames]
        elif C_db.shape[1] < target_frames:
            C_db = np.pad(C_db, ((0,0), (0, target_frames - C_db.shape[1])))
            
        # 6. 정규화 (0~255)
        C_db = (C_db + 80.0) / 80.0 * 255.0
        
        # 7. 차원 추가 (Batch 차원 + Channel 차원)
        # 배치 차원 추가 (N, H, W, C)
        C_db = C_db[..., np.newaxis]
        C_db = C_db[np.newaxis, ...] 
        
        return C_db
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    # 1. 클래스 목록 로드
    classes = get_class_names()
    print(f"총 {len(classes)}개 클래스 탐지")
    
    # 2. 모델 로드
    print("모델 로딩 중...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("모델 로드 완료")
    
    # 3. 테스트 파일 랜덤 선택
    true_label = random.choice(classes)
    class_dir = os.path.join(DATA_DIR, true_label)
    files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
    
    if not files:
        print("파일이 없습니다.")
        return
        
    test_file = random.choice(files)
    file_path = os.path.join(class_dir, test_file)
    
    print(f"\n🎵 테스트 파일: {true_label}/{test_file}")
    
    # 4. 전처리
    input_data = preprocess_audio(file_path)
    if input_data is None:
        return
        
    # 5. 예측
    predictions = model.predict(input_data, verbose=0)
    
    # 6. 결과 분석
    predicted_idx = np.argmax(predictions[0])
    predicted_label = classes[predicted_idx]
    confidence = predictions[0][predicted_idx] * 100
    
    print(f"\n[예측 결과]: {predicted_label} (확신도: {confidence:.2f}%)")
    print(f"[정답]: {true_label}")
    
    if predicted_label == true_label:
        print("결과: 정답")
    else:
        print("결과: 오답")
        
    # Top 3 후보 출력
    print("\n[Top 3 후보]")
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    for idx in top_3_indices:
        print(f"- {classes[idx]}: {predictions[0][idx]*100:.2f}%")

if __name__ == "__main__":
    main()
