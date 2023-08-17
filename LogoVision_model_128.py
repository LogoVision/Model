from tensorflow.keras.models \
    import Sequential
from tensorflow.keras.layers \
    import Conv2D, MaxPooling2D, \
        GlobalAveragePooling2D, Dense
from tensorflow.keras.utils \
    import to_categorical
from cv2 \
    import imread, imshow, waitKey, destroyAllWindows, \
        getRotationMatrix2D, warpAffine, getPerspectiveTransform, warpPerspective
from random \
    import randint
import numpy as np
from LogoVision import imagePath

imagePath = imagePath[:20]

# 이미지 크기 설정
imageHeight, imageWidth = 416, 416

# 클래스의 개수 설정
numClasses = len(imagePath)

# 모델 구성
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(imageHeight, imageWidth, 3)),
    MaxPooling2D((2, 2)), #relu, tanh, sigmoid
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),  # 추가된 밀집층
    Dense(numClasses, activation='softmax')  # 클래스 개수에 맞는 출력층
])

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
for k in range(50):
    for i in range(numClasses):
        print(str(k+1)+"/50", str(i+1)+"/"+str(numClasses), end=" ")
        imgList = []
        labels = to_categorical(np.array([i for j in range(20)]), num_classes=numClasses)  # 레이블을 원-핫 인코딩으로 변환
        img = imread(imagePath[i])
        height, width, channel = img.shape
        for l in range(4):
            matrix = getRotationMatrix2D((width/2, height/2), 90*l, 1)
            dst = warpAffine(img, matrix, (width, height))
            for l in range(5):
                xy = \
                [
                    [randint(0,150),randint(0,150)],\
                    [randint(265,415),randint(0,150)],\
                    [randint(265,415),randint(265,415)],\
                    [randint(0,150),randint(265,415)]
                ]
                imgPoint = np.array(xy,dtype=np.float32)
                dstPoint = np.array([[0,0],[width,0],[width,height],[0,height]], dtype=np.float32)
                matrix = getPerspectiveTransform(imgPoint, dstPoint)
                dst1 = warpPerspective(dst, matrix, (width, height))
                imgList.append(dst1 / 255.0)
        imgList = np.array(imgList)
        # 모델 학습
        model.fit(imgList, labels, epochs=1)

# 왜곡한 이미지로 테스트
testImages = []
testLabels = []
for i in range(numClasses):
    testLabels.append(i)
    img = imread(imagePath[i])
    height, width, channel = img.shape
    xy = \
    [
        [randint(0,150),randint(0,150)],\
        [randint(265,415),randint(0,150)],\
        [randint(265,415),randint(265,415)],\
        [randint(0,150),randint(265,415)]
    ]
    imgPoint = np.array(xy,dtype=np.float32)
    dstPoint = np.array([[0,0],[width,0],[width,height],[0,height]], dtype=np.float32)
    matrix = getPerspectiveTransform(imgPoint, dstPoint)
    dst = warpPerspective(img, matrix, (width, height))
    testImages.append(dst / 255.0)
testImages = np.array(testImages)
testLabels = to_categorical(np.array(testLabels), num_classes=numClasses)

# 모델 평가
test_loss, test_acc = model.evaluate(testImages,testLabels)
print("Test Loss :", test_loss)
print("Test Accuracy :", test_acc)

# 특징 벡터 추출
feature_vectors = model.predict(testImages)
print("Feature Vectors:")
print(feature_vectors)

# # 모델 요약 정보 출력
# model.summary()