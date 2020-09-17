
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
#-------------------------------------#
#       调用摄像头检测
#-------------------------------------#
from keras.layers import Input
from yolo import YOLO
from PIL import Image

import cv2
import time
from keras.layers import Input
from yolo import YOLO
from PIL import Image
import numpy as np

import os

import socket
server = socket.socket()  # AF_INET、SOCK_STREAM
server.bind(('192.168.31.238',6868))   # 将主机号与端口绑定到套接字
server.listen()   # 设置并启动TCP监听器
yolo = YOLO()


cap = cv2.VideoCapture(2)  #创建内置摄像头变量
conn,addr = server.accept()   # 被动接受TCP连接，一直等待连接到达
data=conn.recv(1024)
while True:

    ret,img = cap.read()  #把摄像头获取的图像信息保存之img变量


    frame = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# 转变成Image
    frame = Image.fromarray(np.uint8(frame))
# 进行检测
    tuxiang,zuobiao,label=yolo.detect_image(frame)

    change=zuobiao
    if len(zuobiao)!=0:

        zuobiao=(str(zuobiao[0])+"#"+str(zuobiao[1])+"#"+str(zuobiao[2])+"#"+str(zuobiao[3])+"#"+label).encode("utf-8")

        conn.send(zuobiao)  # 将接收到的数据转为大写在发回给它
    frame = np.array(tuxiang)
# RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    print("zuobiao",( int((change[1]+change[3])/2),int((change[0]+change[2])/2)))
    frame = cv2.putText(frame, "o", ( int((change[1]+change[3])/2),int((change[0]+change[2])/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("video",frame)

    cv2.waitKey(1)
server.close()


yolo.close_session()