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
EPOCHS = 20
LEARNING_RATE = 0.0001 # Fine-tuning을 위해 낮은 학습률 사용
INPUT_SHAPE = (84, 84, 1)
NUM_CLASSES = 24

# ResNet50 기반 모델 생성
def build_model(num_classes):
    # 1. 입력층 (1채널 -> 3채널 변환)
    input_tensor = Input(shape=INPUT_SHAPE)
    x = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(input_tensor)
    
    # 2. ResNet50 (ImageNet 가중치 사용)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(84, 84, 3))
    x = base_model(x)
    
    # 3. Full Fine-tuning (전체 모델 학습)
    base_model.trainable = True
    
    # 4. 분류기 (Classifier)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_tensor, outputs=predictions)
    return model

def main():
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    # 데이터 제너레이터
    train_generator = ChordDataGenerator(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        input_shape=INPUT_SHAPE,
        shuffle=True,
        validation_split=0.2,
        subset='training'
    )
    
    validation_generator = ChordDataGenerator(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        input_shape=INPUT_SHAPE,
        shuffle=False,
        validation_split=0.2,
        subset='validation'
    )
    
    real_num_classes = train_generator.num_classes
    print(f"클래스 개수: {real_num_classes}")
    
    # 모델 생성 및 컴파일
    model = build_model(real_num_classes)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    
    # 학습
    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss', mode='min'),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
    
    print("\n🚀 학습 시작")
    model.fit(
        train_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=validation_generator
    )
    print(f"\n✅ 학습 완료: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
