import os
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm

# ==========================================
# 설정 (Configuration)
# ==========================================
# 데이터가 있는 폴더 위치 (압축 푼 곳)
DATA_ROOT = "c:/AI_PROJECT/data/IDMT-SMT-CHORDS"
# 잘라낸 데이터를 저장할 폴더 위치
OUTPUT_ROOT = "c:/AI_PROJECT/data/processed"
# 오디오 샘플링 레이트 (음질) - 22050Hz는 분석용으로 적당한 표준값입니다.
SAMPLE_RATE = 22050

def parse_lab_file(lab_path):
    """
    .lab 파일을 읽어서 (시작시간, 끝시간, 코드이름) 리스트로 만들어주는 함수입니다.
    마치 '정답지'를 읽어서 컴퓨터가 이해하기 쉬운 표로 만드는 것과 같습니다.
    """
    labels = []
    with open(lab_path, 'r') as f:
        for line in f:
            # 한 줄씩 읽어서 공백을 기준으로 나눕니다.
            # 예: "0.0  2.0  C:maj" -> ['0.0', '2.0', 'C:maj']
            parts = line.strip().split()
            if len(parts) >= 3:
                start = float(parts[0]) # 시작 시간 (초)
                end = float(parts[1])   # 끝나는 시간 (초)
                label = parts[2]        # 코드 이름 (예: C:maj)
                labels.append((start, end, label))
    return labels

def simplify_chord_label(label):
    """
    복잡한 코드 이름을 24개 클래스(Major/Minor)로 단순화합니다.
    예: C:maj7 -> C_maj, A:min/G -> A_min
    Major/Minor가 아닌 것(dim, aug, N 등)은 None을 반환하여 버립니다.
    """
    # 1. 루트 노트와 코드 성질 분리 (예: C:maj7 -> C, maj7)
    if ':' not in label:
        return None
    
    root, rest = label.split(':', 1)
    
    # 2. 전위(/3, /5) 제거 (예: maj7/3 -> maj7)
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
    기타(Guitar)나 피아노(Non-Guitar) 폴더를 처리하는 함수입니다.
    오디오 파일을 불러와서 정답지(lab)대로 싹둑싹둑 자릅니다.
    """
    # 처리할 폴더 경로 만들기
    category_path = os.path.join(DATA_ROOT, subfolder)
    
    # .lab 파일 찾기 (정답지 찾기)
    lab_file = [f for f in os.listdir(category_path) if f.endswith('.lab')][0]
    lab_path = os.path.join(category_path, lab_file)
    
    print(f"[{category_name}] 처리를 시작합니다. 정답지 파일: {lab_file}")
    
    # 정답지 읽어오기
    chord_labels = parse_lab_file(lab_path)
    
    # 폴더 안의 모든 .wav 파일(오디오) 찾기
    wav_files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
    
    # 오디오 파일 하나씩 처리하기 (tqdm은 진행률 바를 보여주는 도구입니다)
    for wav_file in tqdm(wav_files, desc=f"{category_name} 자르는 중"):
        wav_path = os.path.join(category_path, wav_file)
        
        # 1. 오디오 파일 불러오기 (librosa 사용)
        # y: 소리 데이터(숫자들의 배열), sr: 샘플링 레이트
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
        
        # 2. 정답지에 적힌 대로 자르기
        for i, (start, end, label) in enumerate(chord_labels):
            # 24개 클래스로 단순화 (Major/Minor만 남김)
            simple_label = simplify_chord_label(label)
            
            # 원하는 코드가 아니면(None) 건너뜀
            if simple_label is None:
                continue
            
            # 저장할 폴더 만들기 (예: data/processed/C_maj)
            label_dir = os.path.join(OUTPUT_ROOT, simple_label)
            os.makedirs(label_dir, exist_ok=True)
            
            # 시간(초)을 샘플 번호(인덱스)로 변환합니다.
            # 예: 2초 * 22050 = 44100번째 데이터
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            
            # 오디오 길이가 정답지보다 짧으면 건너뜁니다 (에러 방지)
            if start_sample >= len(y):
                continue
                
            # 실제 소리 데이터 자르기 (슬라이싱)
            y_slice = y[start_sample:end_sample]
            
            # 3. 자른 소리 저장하기
            # 파일명 예시: ableton_piano_slice_001.wav
            out_filename = f"{os.path.splitext(wav_file)[0]}_slice_{i:03d}.wav"
            out_path = os.path.join(label_dir, out_filename)
            
            sf.write(out_path, y_slice, sr)

def main():
    # 결과 저장할 폴더가 없으면 만듭니다.
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # Non-Guitar (피아노 등) 폴더가 있으면 처리
    if os.path.exists(os.path.join(DATA_ROOT, "non_guitar")):
        process_category("Non-Guitar (피아노 등)", "non_guitar")
        
    # Guitar (기타) 폴더가 있으면 처리
    if os.path.exists(os.path.join(DATA_ROOT, "guitar")):
        process_category("Guitar (기타)", "guitar")

    print(f"\n모든 작업이 끝났습니다! 결과물은 여기에 있습니다: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
