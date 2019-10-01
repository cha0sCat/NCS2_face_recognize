#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   main.py
@Time    :   2019/10/1 16:37
@Author  :   cha0sCat
@Version :   1.0
@Contact :   ***REMOVED***
@License :   (C)Copyright 2017-2019, cha0sCat
@Desc    :   使用服务端进行人脸搜索
"""
import json
import flask
import numpy as np
import base64
from flask import Flask, jsonify

from utils import *

KNOWN_FACES_PICKLE_PATH = "known_peoples.pkl"

# 首先加载所有已知人脸
known_peoples = loadKnownFacesPickle()
unknown_peoples = {}

app = Flask(__name__)


@app.route("/faceMatch", methods=['POST'])
def face_match():
    """
    使用服务端数据集进行识别
    :return:
    """
    global known_peoples, unknown_peoples
    face_node = pickle.loads(base64.b64decode(flask.request.form.get('face_node', '')))
    FACE_MATCH_THRESHOLD = float(flask.request.form.get('FACE_MATCH_THRESHOLD', ''))

    people_name = faceMatch(face_node, known_peoples, FACE_MATCH_THRESHOLD)

    # 认识这个人
    if people_name:
        logger.info("Match Faces! {}".format(people_name))
        known_peoples[people_name].append(face_node)
        return jsonify({
            "known": True,
            "people_name": people_name
        })

    # 不认识这个人
    else:
        people_name = faceMatch(face_node, unknown_peoples, FACE_MATCH_THRESHOLD)

        # 但是之前见过
        if people_name:
            logger.info("Match Unknown Faces {}".format(people_name))
            unknown_peoples[people_name].append(face_node)
            return jsonify({
                "known": False,
                "exists_before": True,
                "people_name": people_name
            })

        # 根本没见过
        else:
            people_name = "unknownPeople_{}".format(genRandomStrings())
            unknown_peoples[people_name] = []
            unknown_peoples[people_name].append(face_node)
            return jsonify({
                "known": False,
                "exists_before": False,
                "people_name": people_name
            })


@app.route("/update", methods=['POST'])
def updateFaceSet():
    """
    更新服务端数据集
    :return:
    """
    global known_peoples
    # 读入客户端上传的pickle数据集
    new_known_peoples = flask.request.form.get('known_peoples', '')
    new_known_peoples = pickle.loads(base64.b64decode(new_known_peoples))
    # 增加所有新数据
    for known_people_name, known_faces in new_known_peoples.items():
        # 此人已有数据集
        if known_peoples.get(known_people_name):
            known_peoples[known_people_name].extend(known_faces)
            # 去重
            # known_peoples[known_people_name] = list(np.unique(known_peoples[known_people_name]))
            known_peoples[known_people_name] = list(set(known_peoples[known_people_name]))

        else:
            known_peoples[known_people_name] = known_faces
    logger.info("人脸数据集已更新, 当前存储信息:\n{}".format(
        '\n'.join(
            ["{people_name}: {face_node_len}".format(
                people_name=known_people_name,
                face_node_len=len(known_faces))
                for known_people_name, known_faces in known_peoples.items()])
    ))
    return ""


app.run(host="0.0.0.0", port=8088)
