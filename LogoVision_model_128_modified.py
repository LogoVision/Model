from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.utils import to_categorical
from cv2 import imread, imshow, waitKey, destroyAllWindows, getRotationMatrix2D, warpAffine, getPerspectiveTransform, warpPerspective
from random import randint
import numpy as np
from loadImgPath import imagePath

# 이미지 크기 설정
imageHeight, imageWidth = 416, 416

# 클래스의 개수 설정
numClasses = len(imagePath)

# 특징 벡터 차원 설정
numFeatures = 4

# 모델 구성
model = Sequential([
    Conv2D(16, (4, 4), activation='relu', input_shape=(imageHeight, imageWidth, 3)), # 출력 특징 맵의 크기는 414 x 414
    MaxPooling2D((2, 2)), # 출력 특징 맵 207*207
    Conv2D(32, (3, 3), activation='relu'), # 205*205
    MaxPooling2D((5, 5)), # 41*41
    Conv2D(64, (3, 3), activation='relu'), # 39*39
    MaxPooling2D((3, 3)), # 13*13
    Conv2D(128, (4, 4), activation='relu'), # 10*10
    GlobalAveragePooling2D(),
    Dense(128, activation='linear'),
    Dense(64, activation='tanh'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='tanh'),
    Dense(numFeatures, activation='linear') # 특징 벡터를 출력하는 모델이므로 선형 활성화 함수
])

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error', metrics=["accuracy"])  # 거리 기반 손실 함수 사용
model.reset_states()  # 가중치 초기화

# 모델 학습
epochs = 5 # 반복 횟수 설정
for k in range(epochs):
    for i in range(numClasses//5): # class 개수만큼 반복하면서 각각의 클래스 학습
        imgList = []
        feature_vector = []
        feature_vectors = [] # labels 대신에 특징 벡터 사용
        img = []
        for j in range(5):
            try:
                img.append(imread(imagePath[i+j]))
            except:
                break
        # 특징 벡터 생성
        for j in range(5):
            try:
                feature_vector.append(model.predict(np.expand_dims(img[j] / 255.0, axis=0)))
            except:
                break

        for j in range(5):
            try:
                for l in range(4):
                    feature_vectors.append(feature_vector[j])
            except:
                break
        
        # 이미지 증강
        for j in range(5):
            try:
                height, width, channel = img[j].shape
                for l in range(4):
                    # 이미지 회전
                    matrix = getRotationMatrix2D((width/2, height/2), 90*l, 1)
                    dst = warpAffine(img[j], matrix, (width, height))
                    xy = [
                        [randint(0, 150), randint(0, 150)],
                        [randint(265, 415), randint(0, 150)],
                        [randint(265, 415), randint(265, 415)],
                        [randint(0, 150), randint(265, 415)]
                    ]
                    imgPoint = np.array(xy, dtype=np.float32)
                    dstPoint = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
                    matrix = getPerspectiveTransform(imgPoint, dstPoint)
                    dst1 = warpPerspective(dst, matrix, (width, height))
                    imgList.append(dst1 / 255.0) # 회전하고 왜곡한 이미지를 imgList에 추가
            except:
                break
        
        # 학습 데이터 정규화
        imgList = np.array(imgList)
        feature_vectors = np.array(feature_vectors)

        # 모델 학습
        model.fit(imgList, feature_vectors, epochs=1)

# 왜곡한 이미지로 테스트 및 특징 벡터 추출
testImages = []  # 테스트 이미지들의 리스트
testLabels = []  # 테스트 레이블들의 리스트

testClasses = []
for i in range(20):
    a = (numClasses//20)*i
    b = (numClasses//20)*(i+1)
    testClasses.append(randint(a,b-1))

for i in testClasses:
    img = imread(imagePath[i])
    testLabels.append(img / 255.0)
    height, width, channel = img.shape
    xy = [
        [randint(0, 150), randint(0, 150)],
        [randint(265, 415), randint(0, 150)],
        [randint(265, 415), randint(265, 415)],
        [randint(0, 150), randint(265, 415)]
    ]
    imgPoint = np.array(xy, dtype=np.float32)
    dstPoint = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    matrix = getPerspectiveTransform(imgPoint, dstPoint)
    dst = warpPerspective(img, matrix, (width, height))
    testImages.append(dst / 255.0)

testImages = np.array(testImages)
testLabels = np.array(testLabels)

# 테스트 데이터에 대한 특징 벡터 추출
feature_vectors = model.predict(testLabels)
print("Feature Vectors:")
print(feature_vectors)

# 모델 평가
test_loss, test_acc = model.evaluate(testImages, feature_vectors)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)
