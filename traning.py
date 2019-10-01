#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   traning.py
@Time    :   2019/9/30 21:41
@Author  :   cha0sCat
@Version :   1.0
@Contact :   ***REMOVED***
@License :   (C)Copyright 2017-2019, cha0sCat
@Desc    :   对识别后保存的未知人脸进行标识并保存到数据集
"""
import os
import cv2
import pickle

from facerecognize import cv2FaceRecognize as FaceRecognize

network = FaceRecognize()
IMAGE_SAVE_PATH = "images/"
known_peoples = {}
people_names = os.listdir(IMAGE_SAVE_PATH)

for people_name in people_names:
    # 初始化此人的面部特征列表
    known_peoples[people_name] = []
    face_image_names = os.listdir(IMAGE_SAVE_PATH + people_name)
    # 对此人名下的所有图片进行识别
    for face_image_file_name in face_image_names:
        face_image = cv2.imread(IMAGE_SAVE_PATH + people_name + "/" + face_image_file_name)
        face_node = network.runImages(face_image)
        known_peoples[people_name].append(face_node)

with open("known_peoples.pkl", "wb") as f:
    pickle.dump(known_peoples, f)
