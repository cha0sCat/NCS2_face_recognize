#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   test.py
@Time    :   2019/9/30 15:23
@Author  :   cha0sCat
@Version :   1.0
@Contact :   ***REMOVED***
@License :   (C)Copyright 2017-2019, cha0sCat
@Desc    :   None
"""

from main import performOneFrame, initNetwork

import cv2

initNetwork()

frame = cv2.imread("test/pexels-photo.jpg")
performOneFrame(frame)
