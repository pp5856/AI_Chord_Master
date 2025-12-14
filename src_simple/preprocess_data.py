import os
import librosa
import soundfile as sf
from tqdm import tqdm

# 설정
DATA_ROOT = "c:/AI_PROJECT/data/IDMT-SMT-CHORDS"
OUTPUT_ROOT = "c:/AI_PROJECT/data/processed"
SAMPLE_RATE = 22050

def parse_lab_file(lab_path):
    """lab 파일 파싱: (start, end, label) 리스트 반환"""
    labels = []
    with open(lab_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                start = float(parts[0])
                end = float(parts[1])
                label = parts[2]
                labels.append((start, end, label))
    return labels

def simplify_chord_label(label):
    """코드 라벨 단순화 (Major/Minor 24개 클래스)"""
    if ':' not in label:
        return None
    
    root, rest = label.split(':', 1)
    if '/' in rest:
        rest = rest.split('/')[0]
        
    if 'maj' in rest:
        return f"{root}_maj"
    elif 'min' in rest:
        return f"{root}_min"
    else:
        return None

def process_category(category_name, subfolder):
    """카테고리별 오디오 처리 및 슬라이싱"""
    category_path = os.path.join(DATA_ROOT, subfolder)
    lab_file = [f for f in os.listdir(category_path) if f.endswith('.lab')][0]
    lab_path = os.path.join(category_path, lab_file)
    
    print(f"[{category_name}] 처리 시작: {lab_file}")
    chord_labels = parse_lab_file(lab_path)
    wav_files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
    
    for wav_file in tqdm(wav_files, desc=f"{category_name} 처리 중"):
        wav_path = os.path.join(category_path, wav_file)
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
        
        for i, (start, end, label) in enumerate(chord_labels):
            simple_label = simplify_chord_label(label)
            if simple_label is None:
                continue
            
            label_dir = os.path.join(OUTPUT_ROOT, simple_label)
            os.makedirs(label_dir, exist_ok=True)
            
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            
            if start_sample >= len(y):
                continue
                
            y_slice = y[start_sample:end_sample]
            out_filename = f"{os.path.splitext(wav_file)[0]}_slice_{i:03d}.wav"
            out_path = os.path.join(label_dir, out_filename)
            sf.write(out_path, y_slice, sr)

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    if os.path.exists(os.path.join(DATA_ROOT, "non_guitar")):
        process_category("Non-Guitar", "non_guitar")
        
    if os.path.exists(os.path.join(DATA_ROOT, "guitar")):
        process_category("Guitar", "guitar")

    print(f"완료: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
