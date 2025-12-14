import os
import numpy as np
import librosa
import tensorflow as tf
import math

# ==========================================
# 데이터 제너레이터 (Data Generator)
# ==========================================
# AI 모델에게 데이터를 '한 숟가락씩' 떠먹여주는 역할을 합니다.
# 모든 데이터를 한 번에 메모리에 올리면 컴퓨터가 멈출 수 있기 때문에 사용합니다.

class ChordDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, batch_size=32, input_shape=(84, 84, 1), shuffle=True, validation_split=0.0, subset='training'):
        """
        초기화 함수: 제너레이터가 일할 준비를 합니다.
        - data_dir: 처리된 데이터가 있는 폴더 (processed)
        - batch_size: 한 번에 공부할 문제 수 (기본 32개)
        - input_shape: 입력 이미지 크기 (세로 84건반 x 가로 84시간)
        - validation_split: 검증용 데이터 비율 (0.0 ~ 1.0)
        - subset: 'training'(학습용) 또는 'validation'(검증용)
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.subset = subset
        
        # 1. 모든 클래스(코드 이름) 찾기
        # 폴더 이름들을 읽어서 알파벳 순서로 정렬합니다. (A_maj, A_min ...)
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.num_classes = len(self.classes)
        
        # 클래스 이름 <-> 숫자 변환표 만들기
        # 예: {'A_maj': 0, 'A_min': 1, ...}
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 2. 모든 파일 목록 만들기
        # (파일경로, 정답숫자) 쌍을 리스트에 담습니다.
        self.file_list = []
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            files = [f for f in os.listdir(cls_dir) if f.endswith('.npy')]
            for f in files:
                full_path = os.path.join(cls_dir, f)
                label_idx = self.class_to_idx[cls_name]
                self.file_list.append((full_path, label_idx))
        
        # 3. 학습용/검증용 나누기 (Split)
        if validation_split > 0.0:
            # 항상 똑같은 순서로 섞이게 해서, 학습용과 검증용이 섞이지 않게 합니다.
            import random
            self.file_list.sort() # 먼저 정렬하고
            random.Random(42).shuffle(self.file_list) # 고정된 시드(42)로 섞습니다.
            
            split_idx = int(len(self.file_list) * (1 - validation_split))
            
            if subset == 'training':
                self.file_list = self.file_list[:split_idx]
            elif subset == 'validation':
                self.file_list = self.file_list[split_idx:]
            else:
                raise ValueError("subset은 'training' 또는 'validation'이어야 합니다.")
                
        print(f"[{self.subset}] 총 {len(self.file_list)}개의 파일을 찾았습니다. (클래스 개수: {self.num_classes})")
        
        # 데이터 섞기 (공부 순서가 매번 달라야 실력이 늡니다)
        self.on_epoch_end()

    def __len__(self):
        """한 시대(Epoch)에 몇 번 배달을 가야 하는지 계산합니다."""
        return math.ceil(len(self.file_list) / self.batch_size)

    def __getitem__(self, index):
        """
        실제로 데이터를 배달하는 함수입니다.
        index번째 주문(배치)을 만들어서 리턴합니다.
        """
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        
        batch_x = []
        batch_y = []
        
        for i in batch_indexes:
            file_path, label_idx = self.file_list[i]
            
            # [수정됨] 이제 복잡한 요리(CQT 변환)를 안 해도 됩니다!
            # 미리 만들어둔 햇반(.npy)을 전자레인지에 돌리기만 하면 됩니다.
            # 로딩 속도: 0.1초 -> 0.001초 (100배 빨라짐)
            cqt_image = np.load(file_path)
            
            batch_x.append(cqt_image)
            batch_y.append(label_idx)
            
        return np.array(batch_x), tf.keras.utils.to_categorical(batch_y, num_classes=self.num_classes)

    def on_epoch_end(self):
        """한 바퀴 공부가 끝나면 순서를 다시 섞습니다."""
        self.indexes = np.arange(len(self.file_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # _generate_cqt 함수는 이제 필요 없습니다! (삭제)
