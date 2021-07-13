#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   debug.py
@Time    :   2019/10/2 16:53
@Author  :   cha0sCat
@Version :   1.0
@Contact :   ***REMOVED***
@License :   (C)Copyright 2017-2019, cha0sCat
@Desc    :   主程序终端模式 命令接口
"""
import traceback

from functools import wraps


class Status:
    def __init__(self):
        from utils import logger
        self.logger = logger
        self.running = False
        self.last_time_use = {}

    def monitor(self, func):
        """
        监控主程序是否在运行的装饰器
        :return:
        """
        @wraps(func)
        def fake_func(*args, **kwargs):
            self.running = True
            try: func(*args, **kwargs)
            except: print(traceback.format_exc())
            self.running = False

        return fake_func

    def p(self):
        print("Running: {}".format(str(self.running)))
        print(
            "\n" + "\n".join([
                "   {func_name}: {time}s".format(
                    func_name=func_name, time=time
                )
                for func_name, time in self.last_time_use.items()
            ])
        )
