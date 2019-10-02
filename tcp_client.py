#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   tcp_client.py
@Time    :   2019/10/2 17:46
@Author  :   cha0sCat
@Version :   1.0
@Contact :   ***REMOVED***
@License :   (C)Copyright 2017-2019, cha0sCat
@Desc    :   使用TCP与比对服务器进行通讯
"""
from socket import *
from time import ctime


class Client:
    def __init__(self, host: str = "127.0.0.1", ports: list = [8080, 8081, 8082]) -> None:
        # 初始化所有端口到可用状态
        self.port_available = {}
        self.socks = {}
        for port in ports:
            # 与服务器建立连接
            self.socks[port] = socket(AF_INET, SOCK_STREAM)
            self.socks[port].connect((host, port))
            self.port_available[port] = True

    def _findAvailableSock(self):
        """
        寻找一个空闲的sock
        :return:
        """
        while 1:
            for port, is_available in self.port_available.items():
                if is_available:
                    return port

    def sendMessage(self, message: bytes) -> bytes:
        # 取得一个空闲的sock
        port = self._findAvailableSock()
        self.port_available[port] = False
        sock = self.socks[port]
        sock.sendall(message)
        while True:
            data = sock.recv(1024)
        return data