import os
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm



DATA_ROOT = "c:/AI_PROJECT/data/IDMT-SMT-CHORDS"
OUTPUT_ROOT = "c:/AI_PROJECT/data/processed"
# 오디오 샘플링 레이트 (음질) - 22050Hz는 분석용으로 적당한 표준값
SAMPLE_RATE = 22050

def parse_lab_file(lab_path):
    """
    .lab 파일을 파싱하여 (시작시간, 종료시간, 레이블) 리스트 반환
    """
    labels = []
    with open(lab_path, 'r') as f:
        for line in f:
            # 공백 기준 분리
            parts = line.strip().split()
            if len(parts) >= 3:
                start = float(parts[0]) # 시작 시간 
                end = float(parts[1])   # 끝나는 시간
                label = parts[2]        # 코드 이름 
                labels.append((start, end, label))
    return labels

def simplify_chord_label(label):
    """
    복잡한 코드 레이블을 24개 클래스(Major/Minor)로 단순화
    기타 클래스(dim, aug, N 등)는 None 반환
    """
    
    # 1. 루트 노트와 코드 성질 분리 (예: C:maj7 -> C, maj7)
    if ':' not in label:
        return None
    
    root, rest = label.split(':', 1)
    
    # 2. 전위 코드 제거
    if '/' in rest:
        rest = rest.split('/')[0]
        
    # 3. 24개 클래스로 매핑
    # 'maj'가 포함되면 무조건 Major
    if 'maj' in rest:
        return f"{root}_maj"
    # 'min'이 포함되면 무조건 Minor
    elif 'min' in rest:
        return f"{root}_min"
    # 그 외(dim, aug, sus, N 등)는 버림
    else:
        return None

def process_category(category_name, subfolder):
    """
    카테고리별 오디오 파일 처리 및 세그먼테이션
    """
    # 처리할 폴더 경로 만들기
    category_path = os.path.join(DATA_ROOT, subfolder)
    
    # .lab 파일 찾기
    lab_file = [f for f in os.listdir(category_path) if f.endswith('.lab')][0]
    lab_path = os.path.join(category_path, lab_file)
    
    print(f"[{category_name}] 처리를 시작합니다. 정답지 파일: {lab_file}")
    
    chord_labels = parse_lab_file(lab_path)
    
    wav_files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
    
    # 오디오 파일 하나씩 처리하기
    for wav_file in tqdm(wav_files, desc=f"{category_name} 자르는 중"):
        wav_path = os.path.join(category_path, wav_file)
     
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
     
        for i, (start, end, label) in enumerate(chord_labels):
            # 24개 클래스로 단순화 (Major/Minor만 남김)
            simple_label = simplify_chord_label(label)
            
            # 원하는 코드가 아니면(None) 건너뜀
            if simple_label is None:
                continue
            
            label_dir = os.path.join(OUTPUT_ROOT, simple_label)
            os.makedirs(label_dir, exist_ok=True)
            
            # 시간 -> 샘플 인덱스 변환
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            
            if start_sample >= len(y):
                continue
                
            y_slice = y[start_sample:end_sample]
            
            # 3. 오디오 세그먼트 저장
            out_filename = f"{os.path.splitext(wav_file)[0]}_slice_{i:03d}.wav"
            out_path = os.path.join(label_dir, out_filename)
            
            sf.write(out_path, y_slice, sr)

def main():
 
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    

    if os.path.exists(os.path.join(DATA_ROOT, "non_guitar")):
        process_category("Non-Guitar (피아노 등)", "non_guitar")

    if os.path.exists(os.path.join(DATA_ROOT, "guitar")):
        process_category("Guitar (기타)", "guitar")

    print(f"\n모든 작업이 끝났습니다! 결과물은 여기에 있습니다: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
