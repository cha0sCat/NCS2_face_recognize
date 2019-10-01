#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Time    :   2019/9/30 17:30
@Author  :   cha0sCat
@Version :   1.0
@Contact :   ***REMOVED***
@License :   (C)Copyright 2017-2019, cha0sCat
@Desc    :   各种工具
"""
import os
import cv2
import time
import numpy
import random

from math import fabs, sin, radians, cos


def timer(func):
    def fake_func(*args, **kwargs):
        before = time.time()
        result = func(*args, **kwargs)
        after = time.time()-before
        print("perform", str(func), "use {} s".format(after))
        return result
    return fake_func


def faceCompare(face1_output, face2_output, threshold=0.91) -> bool:
    """
    两个人脸对比
    :param face1_output: 人脸1
    :param face2_output: 人脸2
    :param threshold: 最高允许误差
    :return: 是否匹配
    """
    if len(face1_output) != len(face2_output):
        print('length mismatch in faceMatch')
        return False
    total_diff = 0
    # Sum of all the squared differences
    for output_index in range(0, len(face1_output)):
        this_diff = numpy.square(face1_output[output_index] - face2_output[output_index])
        total_diff += this_diff
    # Now take the sqrt to get the L2 difference
    total_diff = numpy.sqrt(total_diff)
    # print(' Total Difference is: ' + str(total_diff))
    if total_diff < threshold:
        # the total difference between the two is under the threshold so
        # the faces match.
        return True

    # differences between faces was over the threshold above so
    # they didn't match.
    return False


@timer
def faceMatch(face1_output, known_peoples: dict, threshold=0.91):
    """
    输入人脸与所有已知人脸比对
    knownPeople格式:
      {
        "张三": [
                  张三的人脸识别结果_1,
                  张三的人脸识别结果_2
                  ]
      }
    :param face1_output: 当前需要对比的人脸
    :param known_peoples: 所有已知人物
    :return: 匹配到的人名
    """
    for known_people_name, known_faces in known_peoples.items():
        for known_face in known_faces:
            if faceCompare(face1_output, known_face, threshold):
                return known_people_name

    return None


def imgRotation(img, degree=90):
    height, width = img.shape[:2]
    # 旋转后的尺寸
    height_new = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    width_new = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

    mat_rotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    mat_rotation[0, 2] += (width_new - width) / 2  # 重点在这步，目前不懂为什么加这步
    mat_rotation[1, 2] += (height_new - height) / 2  # 重点在这步

    img_rotated = cv2.warpAffine(img, mat_rotation, (width_new, height_new), borderValue=(255, 255, 255))

    return img_rotated


def genRandomStrings(length=5):
    return ''.join([random.choice('abcdefghijklmnopqrstuvwxyz') for i in range(length)])


@timer
def saveFaceToFiles(face_image, path):
    if os.path.exists(path):
        if os.path.isdir(path):
            pass
        else:
            os.mkdir(path)
    else:
        os.mkdir(path)
    num = str(len(os.listdir(path))).zfill(3)
    file_path = os.path.join(path, "{}.png".format(num))
    cv2.imwrite(file_path, face_image)
