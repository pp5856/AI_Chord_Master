import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_generator import ChordDataGenerator


# 설정
DATA_DIR = "c:/AI_PROJECT/data/cqt_numpy"
MODEL_SAVE_PATH = "c:/AI_PROJECT/models/chord_model.h5"
BATCH_SIZE = 32
EPOCHS = 20           # 학습 에포크 수
LEARNING_RATE = 0.0001 # 학습률 (미세 조정)
INPUT_SHAPE = (84, 84, 1) # 입력 이미지 형상 (H, W, C)
NUM_CLASSES = 24      # 분류 클래스 수

def build_model(num_classes):
    """ResNet50 기반 코드 분류 모델 정의"""
    # 1. 입력층 (1채널 -> 3채널 변환)
    input_tensor = Input(shape=INPUT_SHAPE)
    x = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(input_tensor)
    
    # 2. ResNet50 모델 로드 (ImageNet 가중치 사용)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(84, 84, 3))
    x = base_model(x)
    
    # 3. 미세 조정 (Fine-Tuning) 활성화
    base_model.trainable = True 
    
    # 4. 분류기 헤드 추가
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # 5. 출력층
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # 모델 생성
    model = Model(inputs=input_tensor, outputs=predictions)
    return model

def main():
    # 1. 모델 저장할 폴더 만들기
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    # 2. 데이터 제너레이터 설정
    # 전체 데이터의 20%는 검증용(Validation)으
    # 학습용 (80%)
    train_generator = ChordDataGenerator(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        input_shape=INPUT_SHAPE,
        shuffle=True,
        validation_split=0.2,
        subset='training'
    )
    
    # 검증용 (20%)
    validation_generator = ChordDataGenerator(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        input_shape=INPUT_SHAPE,
        shuffle=False, # 검증할 때는 굳이 섞을 필요 없음
        validation_split=0.2,
        subset='validation'
    )
    
    # 클래스 개수 자동 파악
    real_num_classes = train_generator.num_classes
    print(f"분류할 클래스 개수: {real_num_classes}개")
    
    # 3. 모델 만들기
    model = build_model(real_num_classes)
    
    # 4. 모델 컴파일 (Adam 옵티마이저)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # 모델 구조 요약 출력
    model.summary()
    
    # 5. 콜백 설정 (ModelCheckpoint, EarlyStopping)
    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss', mode='min'),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
    
    # 6. 학습 시작
    print("\n 학습 시작...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=validation_generator # 검증용 데이터 추가
    )
    
    print(f"\n 학습 완료! 모델 저장됨: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
