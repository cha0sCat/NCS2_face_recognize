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
import asyncio

from camera import Camera
from facedect import FaceDetect
from facerecognize import cv2FaceRecognize as FaceRecognize
from utils import *

IMAGE_SAVE_PATH = "images/"
FACE_MATCH_THRESHOLD = 0.91
CAMERA_ADDRESS = "http://admin:admin@192.168.50.14:8081/"
KNOWN_FACES_PICKLE_PATH = "known_peoples.pkl"

unknown_peoples = {}
known_peoples = loadKnownFacesPickle(KNOWN_FACES_PICKLE_PATH)

face_detect = FaceDetect()
face_recognize = FaceRecognize()
loop = asyncio.get_event_loop()


def initNetwork():
    """
    初始化神经网络，不然第一次检测到人脸的时候加载神经网络需要6s...
    :return:
    """
    frame = cv2.imread("test/elvis.png")
    face_detect.detectFace(frame)
    face_recognize.runImages(frame)


def knownFaceMatchSuccess(people_name, face_image, face_node):
    """
    成功匹配到已知人脸的时候做这些
    :param people_name: 匹配到的人名
    :param face_image: 人脸部位图像
    :param face_node: 人脸识别结果
    :return: None
    """
    global known_peoples
    # 学习+保存
    known_peoples[people_name].append(face_node)
    if len(known_peoples[people_name]) <= 10:
        saveFaceToFiles(face_image, people_name)


def unknownFaceMatchSuccess(people_name, face_image, face_node, exist_before=True):
    """
    成功匹配到未知人脸的时候做这些
    :param people_name: 匹配到的人名
    :param face_image: 人脸部位图像
    :param face_node: 人脸识别结果
    :param exist_before: 未知，但是见过
    :return: None
    """
    global unknown_peoples
    if not exist_before:
        unknown_peoples[people_name] = []
    path = os.path.join(IMAGE_SAVE_PATH, people_name)
    unknown_peoples[people_name].append(face_node)
    saveFaceToFiles(face_image, path)


async def runMatch(face_image, face_node):
    """
    异步进行面部比对
    因为numpy面部比对很费时间，比对一次需要0.007s 40次则需要0.28s
    :param face_image: 人脸图像
    :param face_node: face_recognize.runImages 的识别结果
    :return: None
    """
    people_name = faceMatch(face_node, known_peoples, FACE_MATCH_THRESHOLD)

    # 认识这个人
    if people_name:
        logger.info("Match Faces! {}".format(people_name))
        knownFaceMatchSuccess(people_name, face_image, face_node)

    # 不认识这个人
    else:
        people_name = faceMatch(face_node, unknown_peoples, FACE_MATCH_THRESHOLD)

        # 但是之前见过
        if people_name:
            logger.info("Match Unknown Faces {}".format(people_name))
            unknownFaceMatchSuccess(people_name, face_image, face_node)

        # 根本没见过
        else:
            people_name = "unknownPeople_{}".format(genRandomStrings())
            unknownFaceMatchSuccess(people_name, face_image, face_node, exist_before=False)


@timer
def performOneFrame(frame):
    """
    处理一帧图像
    :param frame: 图像
    :return: None
    """
    # 人脸部分图像
    face_images, scores = face_detect.detectFace(frame, crop=True)

    # 对图像内的每张人脸进行匹配
    for face_image in face_images:
        # 取得人脸标志点 和数据库内的人脸进行匹配
        face_node = face_recognize.runImages(face_image)
        # task = asyncio.create_task(runMatch(face_image, face_node))
        loop.run_until_complete(runMatch(face_image, face_node))


def main():
    initNetwork()
    cam = Camera(CAMERA_ADDRESS)
    while True:
        logger.debug("------------------new frame----------------------")
        # 取得图像和
        frame = cam.get_frame()
        frame = imgRotation(frame, -90)
        performOneFrame(frame)


if __name__ == '__main__':
    main()
