import json
import os
import cv2

# 라벨링 데이터 불러오기
lbPath = "C:\\Users\\user\\Desktop\\sample\\labeled\\labeled_images"
lbDataNames = list(os.walk(lbPath))[0][2]

lbImage = {}
for lbFileName in lbDataNames:
    with open(os.path.join(lbPath, lbFileName), "rb") as lbFile:
        lbImage[lbFileName[:12]] = json.load(lbFile, strict=False)["images"]

# 이미지 경로 불러오기
imagePath = []
basePath = "C:\\Users\\user\\Desktop\\sample\\original\\image"
for imgDir in lbImage.keys():
    dirPath = os.path.join(basePath, imgDir)
    imgFileNames = list(os.walk(dirPath))[0][2]
    for imgFileName in imgFileNames:
        finPath = os.path.join(dirPath, imgFileName)
        imagePath.append(finPath)