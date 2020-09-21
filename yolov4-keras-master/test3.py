import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
# -*- coding: utf-8 -*-
# @Time    : 2020/7/26 18:52
# @Author  : 张茹飞
# @Email   : 1106815482@qq.com
# @File    : myselftest.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
# -------------------------------------#
#       调用摄像头检测
# -------------------------------------#
from keras.layers import Input
from yolo import YOLO
from PIL import Image

import cv2
import time
from keras.layers import Input
from yolo import YOLO
from PIL import Image
import numpy as np
import cv2
import time
import os

import socket

server = socket.socket()  # AF_INET、SOCK_STREAM
# Bind the host number and port to the socket
server.bind(('192.168.31.237', 6868))  # change the host LOCAL ID

server.listen()  # Set up and start the TCP listener
yolo = YOLO()
path = "E:\pycharm\yolov4-keras-master\yolov4-keras-master\VOCdevkit\VOC2007\JPEGImages\\"
path1 = os.listdir(path)
cap = cv2.VideoCapture(0)  # Create built-in camera variables
while True:
    conn, addr = server.accept()
    # Passively accept TCP connections and wait for the connection to arrive
    data = conn.recv(1024)
    # Save the image information obtained by the camera as an img variable
    ret, img = cap.read()
    for i in path1:
        img = cv2.imread(path + i)
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # TO Image
        frame = Image.fromarray(np.uint8(frame))
        # detection
        # tuxiang: image; zuobiao: coordinates
        tuxiang, zuobiao, label = yolo.detect_image(frame)

        change = zuobiao
        if len(zuobiao) != 0:
            zuobiao = (str(zuobiao[0]) + "#" + str(zuobiao[1]) + "#" + str(zuobiao[2]) + "#" + str(
                zuobiao[3]) + "#" + label).encode("utf-8")

            conn.send(zuobiao)  # Convert the received data to uppercase and send it back
        frame = np.array(tuxiang)
        # RGBtoBGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        print("zuobiao", (int((change[1] + change[3]) / 2), int((change[0] + change[2]) / 2)))
        frame = cv2.putText(frame, "o", (int((change[1] + change[3]) / 2), int((change[0] + change[2]) / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("video", frame)

        cv2.waitKey(10)
server.close()

yolo.close_session()
