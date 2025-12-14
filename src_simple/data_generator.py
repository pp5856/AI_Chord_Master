import os
import numpy as np
import tensorflow as tf
import math

class ChordDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, batch_size=32, input_shape=(84, 84, 1), shuffle=True, validation_split=0.0, subset='training'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.subset = subset
        
        # 클래스 로드
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 파일 목록 생성
        self.file_list = []
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            files = [f for f in os.listdir(cls_dir) if f.endswith('.npy')]
            for f in files:
                full_path = os.path.join(cls_dir, f)
                label_idx = self.class_to_idx[cls_name]
                self.file_list.append((full_path, label_idx))
        
        # Train/Validation 분할
        if validation_split > 0.0:
            import random
            self.file_list.sort()
            random.Random(42).shuffle(self.file_list)
            
            split_idx = int(len(self.file_list) * (1 - validation_split))
            if subset == 'training':
                self.file_list = self.file_list[:split_idx]
            elif subset == 'validation':
                self.file_list = self.file_list[split_idx:]
                
        print(f"[{self.subset}] 파일 수: {len(self.file_list)} (클래스: {self.num_classes})")
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.file_list) / self.batch_size)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch_x = []
        batch_y = []
        
        for i in batch_indexes:
            file_path, label_idx = self.file_list[i]
            # .npy 파일 로드 (CQT 이미지)
            cqt_image = np.load(file_path)
            batch_x.append(cqt_image)
            batch_y.append(label_idx)
            
        return np.array(batch_x), tf.keras.utils.to_categorical(batch_y, num_classes=self.num_classes)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.file_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)
