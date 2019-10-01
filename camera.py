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
import time

from concurrent.futures import ThreadPoolExecutor


class Camera:
    def __init__(self, videoServerPath, high_frame_mode=False):
        self.video_capture = cv2.VideoCapture(videoServerPath)
        self.videoServerPath = videoServerPath
        self.high_frame_mode = high_frame_mode
        if high_frame_mode:
            self.executor = ThreadPoolExecutor(max_workers=5)
            self.executor.submit(self._enableHighFrameMode)
            time.sleep(0.5)

    def reload_stream(self):
        """
        有时候有bug  需要重置链接
        :return:
        """
        self.video_capture = cv2.VideoCapture(self.videoServerPath)  # 链接视频
        return

    def _get_frame(self):
        """
        获取一帧图像
        :return: 一帧图像
        """
        ret, frame = self.video_capture.read()
        return frame

    def get_frame(self):
        if self.high_frame_mode:
            return self.frame
        else:
            return self._get_frame()

    def _enableHighFrameMode(self):
        while True:
            self.frame = self._get_frame()
            time.sleep(0.01)
