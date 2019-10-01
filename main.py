#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   main.py
@Time    :   2019/9/30 14:41
@Author  :   cha0sCat
@Version :   1.0
@Contact :   ***REMOVED***
@License :   (C)Copyright 2017-2019, cha0sCat
@Desc    :   基于NCS2的人脸识别系统
DetectFace (retail-0004)
|
for each face in Detected
|
Landmarks
|
compare with known Faces
"""
import os

from camera import Camera
from facedect import FaceDetect
from facerecognize import cv2FaceRecognize as FaceRecognize
from utils import *

IMAGE_SAVE_PATH = "images/"
FACE_MATCH_THRESHOLD = 0.91


def main():
    cam = Camera("http://admin:admin@192.168.50.14:8081/")
    face_detect = FaceDetect()
    face_recognize = FaceRecognize()
    known_peoples = {}
    unknown_peoples = {}
    while True:
        print("new frame")
        # 取得图像和
        frame = cam.get_frame()
        frame = imgRotation(frame, -90)
        # 人脸部分图像
        face_images, scores = face_detect.detectFace(frame, crop=True)

        # 对图像内的每张人脸进行匹配
        for face_image in face_images:
            # 取得人脸标志点
            face_node = face_recognize.runImages(face_image)
            # 和数据库内的人脸进行匹配
            people_name = faceMatch(face_node, known_peoples, FACE_MATCH_THRESHOLD)

            # 匹配成功
            if people_name:
                print("Match Faces!", people_name)
                # 学习+保存
                known_peoples[people_name].append(face_node)
                if len(known_peoples[people_name]) <= 10:
                    saveFaceToFiles(face_image, people_name)

            # 匹配失败
            else:
                people_name = faceMatch(face_node, unknown_peoples, FACE_MATCH_THRESHOLD)

                # 虽然匹配失败，但是之前见过
                if people_name:
                    print("Match Unknown Faces", people_name)

                # 根本没见过
                else:
                    people_name = "unknownPeople_{}".format(genRandomStrings())
                    unknown_peoples[people_name] = []

                # 学习+保存
                path = os.path.join(IMAGE_SAVE_PATH, people_name)
                unknown_peoples[people_name].append(face_node)
                saveFaceToFiles(face_image, path)


if __name__ == '__main__':
    main()
