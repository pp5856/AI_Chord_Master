import os
import numpy as np
import librosa
import tensorflow as tf
import math

# 데이터 제너레이터 (Data Generator)


class ChordDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, batch_size=32, input_shape=(84, 84, 1), shuffle=True, validation_split=0.0, subset='training'):
        """
        데이터 제너레이터 초기화
        - data_dir: 전처리된 데이터 디렉토리
        - batch_size: 배치 크기
        - input_shape: 입력 이미지 크기
        - validation_split: 검증 데이터 비율
        - subsets: 'training' 또는 'validation'
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.subset = subset
        
        # 1. 클래스 목록 탐색 및 정렬
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.num_classes = len(self.classes)
        
        # 클래스명 -> 인덱스 매핑 생성
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 2. 전체 파일 목록 생성
        self.file_list = []
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            files = [f for f in os.listdir(cls_dir) if f.endswith('.npy')]
            for f in files:
                full_path = os.path.join(cls_dir, f)
                label_idx = self.class_to_idx[cls_name]
                self.file_list.append((full_path, label_idx))
        
        # 3. 데이터 분할 (Train/Validation)
        if validation_split > 0.0:
            # 일관된 분할을 위한 시드 고정 및 셔플
            import random
            self.file_list.sort() # 먼저 정렬하고
            random.Random(42).shuffle(self.file_list) # 고정된 시드(42)로 섞기
            
            split_idx = int(len(self.file_list) * (1 - validation_split))
            
            if subset == 'training':
                self.file_list = self.file_list[:split_idx]
            elif subset == 'validation':
                self.file_list = self.file_list[split_idx:]
            else:
                raise ValueError("subset은 'training' 또는 'validation'이어야 합니다.")
                
        print(f"[{self.subset}] 총 {len(self.file_list)}개의 파일을 찾았습니다. (클래스 개수: {self.num_classes})")
        
        # 초기 데이터 셔플 수행
        self.on_epoch_end()

    def __len__(self):
        """한 시대(Epoch)에 몇 번 배달을 가야 하는지 계산합니다."""
        return math.ceil(len(self.file_list) / self.batch_size)

    def __getitem__(self, index):
        """
        """배치 단위 데이터 생성 및 반환"""
        """
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        
        batch_x = []
        batch_y = []
        
        for i in batch_indexes:
            file_path, label_idx = self.file_list[i]
            
            # 전처리된 CQT 데이터 로드 (.npy)
            cqt_image = np.load(file_path)
            
            batch_x.append(cqt_image)
            batch_y.append(label_idx)
            
        return np.array(batch_x), tf.keras.utils.to_categorical(batch_y, num_classes=self.num_classes)

    def on_epoch_end(self):
        """에포크 종료 시 데이터 셔플"""
        self.indexes = np.arange(len(self.file_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

