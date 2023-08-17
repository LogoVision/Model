from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
import cv2
import numpy as np

# 기본 설정 초기화
imageHeight, imageWidth = 416, 416  # 이미지 크기 설정
numClasses = 2  # 클래스의 개수를 2로 설정

# 이미지 불러오기
imagePath = "C:\\Users\\user\\Desktop\\sample\\original\\image\\010102_TOTAL\\010102_339715.jpg"
imagePath2 = "C:\\Users\\user\\Desktop\\sample\\original\\image\\010102_TOTAL\\010102_1111172.jpg"
image = cv2.imread(imagePath)
image2 = cv2.imread(imagePath2)
image = image / 255.0
image2 = image2 / 255.0
imagesRotated = []
imagesRotated2 = []

# 90도씩 회전시켜 데이터 증강
for angle in [0, 90, 180, 270]:
    rotatedImage = np.rot90(image, k=angle // 90)
    imagesRotated.append(rotatedImage)
    rotatedImage = np.rot90(image2, k=angle // 90)
    imagesRotated2.append(rotatedImage)

# 이미지 리스트를 NumPy 배열로 변환
imagesRotated = np.array(imagesRotated+imagesRotated2)

# 모델 구성
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(imageHeight, imageWidth, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='sigmoid'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='sigmoid'),
    MaxPooling2D((2,2)),
    GlobalAveragePooling2D(),
])

# 중간층을 마지막 레이어로 설정
model.add(Dense(numClasses, activation='softmax'))

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 라벨 데이터 준비
labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # 클래스 개수에 맞게 라벨 설정 (이진 분류)

# 모델 학습
model.fit(imagesRotated, labels, epochs=10)  # 예시: 10 에포크로 학습

# 좌우반전한 이미지로 테스트
testImagesFlipped = []
rotatedImage = np.rot90(image, k=3)
testImagesFlipped.append(rotatedImage)
rotatedImage = np.rot90(image2, k=3)
testImagesFlipped.append(rotatedImage)

# 테스트 이미지 리스트를 NumPy 배열로 변환
testImagesFlipped = np.array(testImagesFlipped)

# 테스트 라벨 데이터 준비
testLabelsFlipped = np.array([0, 1])  # 테스트 이미지의 실제 라벨 (이진 분류)

# 모델 평가
test_loss_flipped, test_acc_flipped = model.evaluate(testImagesFlipped, testLabelsFlipped)
print("Test Loss :", test_loss_flipped)
print("Test Accuracy :", test_acc_flipped)

# 특징 벡터 추출
feature_vectors = model.predict(testImagesFlipped)

print("Feature Vectors:")
print(feature_vectors)