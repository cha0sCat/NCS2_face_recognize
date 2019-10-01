#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   facerecognize.py
@Time    :   2019/9/30 17:30
@Author  :   cha0sCat
@Version :   1.0
@Contact :   ***REMOVED***
@License :   (C)Copyright 2017-2019, cha0sCat
@Desc    :   单面孔人脸辨识
"""
from openvino.inference_engine import IENetwork, IECore
from utils import timer
import sys
import numpy
import cv2
import os


class FaceRecognize:
    def __init__(self,
                 bin_path="models/20180408-102900.bin",
                 xml_path="models/20180408-102900.xml",
                 DEVICE="MYRIAD"):
        ie = IECore()
        net = IENetwork(model=xml_path, weights=bin_path)
        input_blob = next(iter(net.inputs))
        output_blob = next(iter(net.outputs))

        exec_net = ie.load_network(network=net, device_name=DEVICE)
        n, c, network_input_h, network_input_w = net.inputs[input_blob].shape

        self.input_blob = input_blob
        self.output_blob = output_blob
        self.exec_net = exec_net
        self.network_input_h = network_input_h
        self.network_input_w = network_input_w

    def _preprocessImage(self, frame):
        """
        将正常图片转换为适合神经网络的图片格式
        :param frame: cv2读入的图片原图
        :return:
        """
        preprocessed_image = cv2.resize(frame, (self.network_input_w, self.network_input_h))
        preprocessed_image = numpy.transpose(preprocessed_image)
        preprocessed_image = numpy.reshape(preprocessed_image, (1, 3, self.network_input_w, self.network_input_h))

        return preprocessed_image

    def _runInference(self, image_to_classify):
        """
        进行识别
        :param image_to_classify: 经过_preprocessed_image处理的图像
        :return: 识别结果
        """
        results = self.exec_net.infer({self.input_blob: image_to_classify})

        return results[self.output_blob].flatten()

    def runImages(self, frame):
        """
        对人脸进行辨识
        :param frame: cv2的图像
        :return: 人脸标志点
        """
        image_to_classify = self._preprocessImage(frame)
        nodes = self._runInference(image_to_classify)

        return nodes


class cv2FaceRecognize:
    def __init__(self,
                 bin_path="models/20180408-102900.bin",
                 xml_path="models/20180408-102900.xml",
                 network_input_h=160,
                 network_input_w=160):
        net = cv2.dnn.readNet(xml_path, bin_path)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
        self.net = net
        self.network_input_h = network_input_h
        self.network_input_w = network_input_w

    def _preprocessImage(self, frame):
        """
        将正常图片转换为适合神经网络的图片格式
        :param frame: cv2读入的图片原图
        :return:
        """
        blob = cv2.dnn.blobFromImage(frame, size=(self.network_input_w, self.network_input_h))

        return blob

    def _runInference(self, blob):
        """
        进行识别
        :param image_to_classify: 经过_preprocessed_image处理的图像
        :return: 识别结果
        """
        self.net.setInput(blob)
        out = self.net.forward()

        return out

    @timer
    def runImages(self, frame):
        """
        对人脸进行辨识
        :param frame: cv2的图像
        :return: 人脸标志点
        """
        blob = self._preprocessImage(frame)
        nodes = self._runInference(blob)

        return tuple(nodes[0])


if __name__ == '__main__':
    #net = cv2FaceRecognize()
    #face1 = net.runImages(cv2.imread("elvis.png"))[0]
    #face2 = net.runImages(cv2.imread("valid_face.png"))[0]

    net2 = FaceRecognize()
    face1 = net2.runImages(cv2.imread("elvis.png"))
    face2 = net2.runImages(cv2.imread("valid_face.png"))
    # print(net2.runImages(cv2.imread("elvis.png")))
    # print(len(net2.runImages(cv2.imread("elvis.png"))))

    from utils import faceCompare
    print(faceCompare(face1, face2))
