import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_generator import ChordDataGenerator

# ==========================================
# 설정 (Configuration)
# ==========================================
DATA_DIR = "c:/AI_PROJECT/data/cqt_numpy"
MODEL_SAVE_PATH = "c:/AI_PROJECT/models/chord_model.h5"
BATCH_SIZE = 32
EPOCHS = 20           # 미세 조정은 섬세하게 오래 해야 하므로 횟수를 늘립니다.
LEARNING_RATE = 0.0001 # 뇌를 녹였으니, 지식이 망가지지 않게 아주 조심스럽게(낮은 학습률) 공부합니다.
INPUT_SHAPE = (84, 84, 1) # 입력 이미지 크기 (세로, 가로, 흑백1채널)
NUM_CLASSES = 24      # 분류할 코드 개수 (Major 12개 + Minor 12개 등 데이터에 따라 자동 결정됨)

def build_model(num_classes):
    """
    AI 모델(뇌)을 조립하는 함수입니다.
    ResNet50이라는 천재의 뇌를 빌려와서(전이 학습), 우리 목적에 맞게 개조합니다.
    """
    # 1. 입력층 정의 
    # 흑백 이미지(1채널)를 받지만, ResNet은 컬러(3채널)를 좋아하므로 3장 겹쳐서 흉내냅니다.
    input_tensor = Input(shape=INPUT_SHAPE)
    x = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(input_tensor) # 1채널 -> 3채널로 뻥튀기
    
    # 2. 베이스 모델 불러오기 
    # include_top=False: "마지막 분류기(개/고양이 맞추는 부분)는 떼고 가져와"
    # weights='imagenet': "이미지넷 데이터로 미리 공부한 지식을 가져와"
    # 주의: input_tensor를 직접 넣으면 에러가 날 수 있어서, input_shape만 지정하고 따로 연결합니다.
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(84, 84, 3))
    x = base_model(x)
    
    # 3. 베이스 모델 녹이기 (Fine-Tuning)
    # "천재의 지식을 우리 문제에 맞게 조금만 수정하자"
    # 처음부터 다 학습하면 너무 오래 걸리지만, 이미 1차 학습을 했거나 데이터가 충분하면
    # 전체를 미세하게 조정(Fine-Tuning)하는 것이 성능이 훨씬 좋습니다.
    base_model.trainable = True 
    
    # 4. 우리만의 분류기 붙이기 (머리)
    # x = base_model.output # 위에서 이미 연결했으므로 이 줄은 필요 없음
    x = GlobalAveragePooling2D()(x) # 이미지 특징을 압축 (평균내기)
    x = Dense(1024, activation='relu')(x) # 생각할 뉴런 1024개 추가
    x = Dropout(0.5)(x) # 과외 공부 너무 많이 해서 멍청해지는 것(과적합) 방지
    
    # 5. 최종 출력층 (입)
    # num_classes개의 확률을 뱉어냄 (예: C코드일 확률 80%, Am일 확률 5%...)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # 모델 조립 완료
    model = Model(inputs=input_tensor, outputs=predictions)
    return model

def main():
    # 1. 모델 저장할 폴더 만들기
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    # 2. 데이터 제너레이터 준비 (급식 당번)
    # 전체 데이터의 20%는 검증용(Validation)으로 따로 빼둡니다.
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
    
    # 4. 학습 방법 설정 (Compile)
    # optimizer='adam': 가장 성능 좋은 공부법
    # loss='categorical_crossentropy': 객관식 문제 틀린 정도를 계산하는 법
    # metrics=['accuracy']: "몇 점 맞았니?" (정확도)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # 모델 구조 요약 출력
    model.summary()
    
    # 5. 학습 도우미 설정 (Callbacks)
    callbacks = [
        # 시험 잘 볼 때마다 저장해! (기준: val_loss가 낮을수록 좋음)
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss', mode='min'),
        # 성적이 더 안 오르면 그만해! (시간 낭비 방지)
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
    
    # 6. 진짜 학습 시작! (Fit)
    print("\n🚀 학습을 시작합니다! (시간이 좀 걸립니다)")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=validation_generator # 검증용 데이터 추가
    )
    
    print(f"\n✅ 학습 완료! 모델이 저장되었습니다: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
