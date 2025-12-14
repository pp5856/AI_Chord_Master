import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from data_generator import ChordDataGenerator

# 설정
DATA_DIR = "c:/AI_PROJECT/data/cqt_numpy"
MODEL_PATH = "c:/AI_PROJECT/models/chord_model.h5"
BATCH_SIZE = 32
INPUT_SHAPE = (84, 84, 1)

def main():
    # 1. 모델 파일 로드
    if not os.path.exists(MODEL_PATH):
        print(f"모델 파일이 없습니다: {MODEL_PATH}")
        return

    print("모델을 불러오는 중...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("모델 로드 완료")

    # 2. 검증 데이터 제너레이터 초기화

    validation_generator = ChordDataGenerator(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        input_shape=INPUT_SHAPE,
        shuffle=False, 
        validation_split=0.2,
        subset='validation'
    )

    print(f"\n검증 데이터 개수: {len(validation_generator.file_list)}")
    
    # 3. 기본 성능 평가 (Loss, Accuracy)
    print("\n모델 평가 중...")
    loss, accuracy = model.evaluate(validation_generator)
    print(f"\n[기본 평가 결과]")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy*100:.2f}%")

    # 4. 상세 평가 수행
    print("\n검증 데이터 예측 수행 중...")
    
    # 예측값 얻기
    predictions = model.predict(validation_generator)
    y_pred = np.argmax(predictions, axis=1)
    
    # 실제 정답 얻기
    y_true = [item[1] for item in validation_generator.file_list]
    
    # 클래스 이름 가져오기
    class_names = validation_generator.classes

    # 5. 분류 리포트 출력
    print("\n[상세 분류 보고서]")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # 6. 혼동 행렬(Confusion Matrix) 출력
    print("\n[혼동 행렬]")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # 혼동 행렬 히트맵 저장
    try:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        save_path = 'c:/AI_PROJECT/docs/confusion_matrix.png'
        plt.savefig(save_path)
        print(f"\n혼동 행렬 이미지가 저장되었습니다: {save_path}")
    except Exception as e:
        print(f"\n이미지 저장 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
