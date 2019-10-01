#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   camera.py
@Time    :   2019/9/30 19:05
@Author  :   cha0sCat
@Version :   1.0
@Contact :   ***REMOVED***
@License :   (C)Copyright 2017-2019, cha0sCat
@Desc    :   对接摄像头
"""
import cv2


class Camera:
    def __init__(self, videoServerPath):
        self.video_capture = cv2.VideoCapture(videoServerPath)
        self.videoServerPath = videoServerPath

    def reload_stream(self):
        """
        有时候有bug  需要重置链接
        :return:
        """
        self.video_capture = cv2.VideoCapture(self.videoServerPath)  # 链接视频
        return

    def get_frame(self):
        """
        获取一帧图像
        :return: 一帧图像
        """
        ret, frame = self.video_capture.read()
        return frame
