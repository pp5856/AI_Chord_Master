import os
import numpy as np
import librosa
import concurrent.futures
from tqdm import tqdm

# 설정
DATA_DIR = "c:/AI_PROJECT/data/processed"
OUTPUT_DIR = "c:/AI_PROJECT/data/cqt_numpy"
INPUT_SHAPE = (84, 84, 1)

def process_file(args):
    file_path, save_path = args
    
    try:
        # 1. 오디오 로드
        y, sr = librosa.load(file_path, sr=22050)
        
        # 2. 길이 맞추기 (Padding/Cropping)
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
        
        # 5. 크기 맞추기 (Safety Check)
        if C_db.shape[1] > target_frames:
            C_db = C_db[:, :target_frames]
        elif C_db.shape[1] < target_frames:
            C_db = np.pad(C_db, ((0,0), (0, target_frames - C_db.shape[1])))
            
        # 6. 정규화 (0~255)
        C_db = (C_db + 80.0) / 80.0 * 255.0
        
        # 7. 차원 추가 (84, 84, 1)
        C_db = C_db[..., np.newaxis]
        
        # 8. 저장 (.npy)
        np.save(save_path, C_db.astype(np.float32))
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    tasks = []
    
    # 모든 클래스 폴더 순회
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    
    print(f"총 {len(classes)}개 클래스를 처리합니다.")
    
    for cls in classes:
        src_cls_dir = os.path.join(DATA_DIR, cls)
        dst_cls_dir = os.path.join(OUTPUT_DIR, cls)
        
        os.makedirs(dst_cls_dir, exist_ok=True)
        
        files = [f for f in os.listdir(src_cls_dir) if f.endswith('.wav')]
        
        for f in files:
            src_path = os.path.join(src_cls_dir, f)
            dst_path = os.path.join(dst_cls_dir, f.replace('.wav', '.npy'))
            
            # 이미 변환된 파일은 건너뛰기 (이어하기 기능)
            if not os.path.exists(dst_path):
                tasks.append((src_path, dst_path))
    
    print(f"변환할 파일 개수: {len(tasks)}개")
    
    # 병렬 처리 (CPU 코어 다 쓰기)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, tasks), total=len(tasks), desc="CQT 변환 중"))
        
    print("완료!")

if __name__ == "__main__":
    main()
