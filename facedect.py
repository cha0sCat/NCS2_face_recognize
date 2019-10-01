#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   AI.py
@Time    :   2019/9/30 14:44
@Author  :   cha0sCat
@Version :   1.0
@Contact :   ***REMOVED***
@License :   (C)Copyright 2017-2019, cha0sCat
@Desc    :   人脸检测
"""
import cv2

from utils import timer


# 基于 face-detection-retail-0004 模型的人脸检测
# 0004模型单帧时间0.037
# 0001adas模型单帧时间0.2289
# name: "input" , shape: [1x3x300x300] - An input image in the format [BxCxHxW], where:
#  + B - batch size
#  + C - number of channels
#  + H - image height
#  + W - image width
# Expected color order - BGR.
class FaceDetect:
    def __init__(self,
                 bin_path="models/face-detection-retail-0004.bin",
                 xml_path="models/face-detection-retail-0004.xml",
                 network_input_h=300,
                 network_input_w=300):
        net = cv2.dnn.readNet(xml_path, bin_path)
        self.net = net
        self.network_input_h = network_input_h
        self.network_input_w = network_input_w
        self.target_device = cv2.dnn.DNN_TARGET_MYRIAD

    @timer
    def detectFace(self, frame, score=0.5, crop=False):
        """
        检测目标图像中的人脸， 返回人脸坐标
        :param crop: 是否回传图片
        :param frame: 原图像输入
        :param score: 置信度过滤
        :param target_device: 计算设备(默认使用NCS2)
        :return: 人脸及对应置信度
        """
        net = self.net
        # Specify target device.
        net.setPreferableTarget(self.target_device)

        # Prepare input blob and perform an inference.
        blob = cv2.dnn.blobFromImage(frame, size=(self.network_input_w, self.network_input_h), ddepth=cv2.CV_8U)
        net.setInput(blob)
        out = net.forward()
        image_w = frame.shape[1]
        image_h = frame.shape[0]

        faces, scores, cropImgs = [], [], []
        for detection in out.reshape(-1, 7):
            confidence = float(detection[2])

            if confidence > score:
                box_left = int(detection[3] * image_w)
                box_top = int(detection[4] * image_h)
                box_right = int(detection[5] * image_w)
                box_bottom = int(detection[6] * image_h)
                faces.append((box_left, box_top, box_right, box_bottom))
                scores.append(confidence)
                # 要求回传图片
                if crop:
                    cropImgs.append(frame[box_top:box_bottom, box_left:box_right])
            else:
                continue
        if crop:
            return cropImgs, scores
        else:
            return faces, scores


if __name__ == '__main__':
    network = FaceDetect()
    from facerecognize import FaceRecognize
    n2 = FaceDetect()
    print(network.detectFace(cv2.imread("elvis.png")))
