#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   tcp_client.py
@Time    :   2019/10/2 17:46
@Author  :   cha0sCat
@Version :   1.0
@Contact :   ***REMOVED***
@License :   (C)Copyright 2017-2019, cha0sCat
@Desc    :   使用TCP与树莓派进行通讯
"""
from socket import *
from time import ctime


class Server:
    def __init__(self, host: str = "0.0.0.0", ports: list = [8080, 8081, 8082]) -> None:
        # 初始化所有端口到可用状态
        self.port_available = {}
        self.socks = {}
        for port in ports:
            self.socks[port] = socket(AF_INET, SOCK_STREAM)
            self.socks[port].bind((host, port))
            self.socks[port].listen(5)
            self.port_available[port] = True