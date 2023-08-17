from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.utils import to_categorical #
from cv2 import imread, imshow, waitKey, destroyAllWindows, getRotationMatrix2D, warpAffine, getPerspectiveTransform, warpPerspective
from random import randint
import numpy as np
from LogoVision import imagePath

imagePath = imagePath[:20]

# 이미지 크기 설정
imageHeight, imageWidth = 416, 416

# 클래스의 개수 설정
numClasses = len(imagePath)

# 특징 벡터 차원 설정 : 차원이 높을수록 더 자세한 정보 저장 가능하지만 과적합 가능성이 올라감.
numFeatures = 3

# 모델 구성 Sequential : 각층을 순차적으로 실행하는 모델
model = Sequential([

    # Conv2D : 2차원 입력을 받아 특징 맵을 만드는 층.. 아직 작동 원리는 잘 몰라서 찾아보는 중. 가장 먼저 주는 매개변수가 작을수록 간단한 특징을, 클수록 추상적이고 세세한 특징을 인식
    Conv2D(16, (3, 3), activation='relu', input_shape=(imageHeight, imageWidth, 3)),

    # MaxPooling2D : 풀링층 -> (2, 2)를 매개변수로 주면 2x2 범위에서 대푯값을 추출해서 차원의 개수를 줄임
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # GlobalAveragePooling2D : 2차원 입력을 받아 1차원 입력으로 바꾸는 층 -> 2차원 특징 맵을 1차원 특징 벡터로 변환
    GlobalAveragePooling2D(),

    # Dense : 밀집층 -> 이전 층의 모든 출력과 연결된 층, 출력의 개수 조절 가능
    Dense(128, activation='relu'),

    # linear : 최종 출력으로 선형 활성화 함수를 사용하여 특징 벡터를 출력하는데 용이하도록 함.
    # 선형 활성화 함수가 좋은 이유는 잘 몰라서 찾아보는 중..
    Dense(numFeatures, activation='linear')
])

# 모델 컴파일
# optimizer(학습 알고리즘) adam : 학습률을 자동으로 조절하여 각각의 파라미터에 가중치 업데이트, 빠른 학습 속도와 안정성
# optimizer는 이후에 nadam(adam과 nesterov 기울기 갱신(경사하강법의 일종)의 아이디어를 합친 방법)으로 바꿀 예정
# loss(손실 함수) mean squared error(평균 제곱 오차) : 예측값과 실제 타깃값 사이의 차이를 계산하여 모델의 성능을 평가하는데 사용하는 함수
model.compile(optimizer='nadam', loss='mean_squared_error', metrics=['accuracy'])  # 거리 기반 손실 함수 사용

# 모델 학습
# 노트북에서 효과적으로 돌아가도록 적은 양의 데이터를 가지고 와서 여러 번 반복해서 학습하는 방법 사용.
# 좋은 컴퓨터에서 사용한다면 한번에 학습하는 데이터의 양을 늘려도 될 듯..?
epochs = 5 # 반복 횟수 설정
for k in range(epochs):

    # 클래스의 개수만큼 반복해서 학습.
    for i in range(numClasses):
        print(str(k+1)+"/"+str(epochs), str(i+1)+"/"+str(numClasses), end="\n") # 몇 번째 학습인지, 어떤 클래스를 학습시키는지 시각화
        imgList = [] # 이미지를 저장할 리스트
        feature_vectors = [] # 특징 벡터(라벨)를 저장할 리스트

        # 이미지 불러오기(cv2 라이브러리의 imread)
        img = imread(imagePath[i])

        # 특징 벡터 생성 및 저장
        # 같은 이미지를 회전시키거나 왜곡해서 증강시키므로 하나의 대표 이미지에서 추출한 특징 벡터를 사용
        feature_vector = model.predict(np.expand_dims(img / 255.0, axis=0))
        for l in range(20):
            feature_vectors.append(feature_vector)

        # 이미지 증강
        # 테스트를 위해 쉬운 기법만 사용 -> 90도 회전, 왜곡
        height, width, channel = img.shape
        
        # 이미지 회전을 4번 반복
        for l in range(4):
            matrix = getRotationMatrix2D((width/2, height/2), 90*l, 1)
            dst = warpAffine(img, matrix, (width, height))

            # 회전한 이미지를 5번 왜곡
            for l in range(5):
                xy = [
                    [randint(0, 200), randint(0, 200)],
                    [randint(215, 415), randint(0, 200)],
                    [randint(215, 415), randint(215, 415)],
                    [randint(0, 200), randint(215, 415)]
                ]
                imgPoint = np.array(xy, dtype=np.float32)
                dstPoint = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
                matrix = getPerspectiveTransform(imgPoint, dstPoint)
                dst1 = warpPerspective(dst, matrix, (width, height))

                # 왜곡한 이미지 저장
                imgList.append(dst1 / 255.0) # 255로 나누는 이유는 각 픽셀의 값을 0에서 1 사이로 정규화하기 위함
        
        # 이미지 및 특징 벡터를 NumPy에서 제공하는 배열로 변환
        # tensorflow에서는 np.array로 변환한 배열을 사용하기 때문
        imgList = np.array(imgList)
        feature_vectors = np.array(feature_vectors)

        # 모델 학습
        model.fit(imgList, feature_vectors, epochs=1)

# 왜곡한 이미지로 테스트 및 특징 벡터 추출
testImages = []  # 테스트 이미지들의 리스트
testLabels = []  # 테스트 라벨 리스트(정답이 되는 특징 벡터를 저장)

# 각각의 클래스에서 이미지를 불러와 임의로 왜곡
for i in range(numClasses):
    img = imread(imagePath[i])

    # 이미지를 왜곡하기 전의 이미지를 testLabels에 저장
    testLabels.append(img / 255.0)
    height, width, channel = img.shape
    xy = [
        [randint(0, 200), randint(0, 200)],
        [randint(215, 415), randint(0, 200)],
        [randint(215, 415), randint(215, 415)],
        [randint(0, 200), randint(215, 415)]
    ]
    imgPoint = np.array(xy, dtype=np.float32)
    dstPoint = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    matrix = getPerspectiveTransform(imgPoint, dstPoint)
    dst = warpPerspective(img, matrix, (width, height))

    # 왜곡한 이미지를 testImages에 저장
    testImages.append(dst / 255.0)

# 테스트 이미지를 np.array로 변환
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