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

# Detection for Local files of images only
path = "C:\\Users\\chunp\\Desktop\\fisted_samples\\"
yolo = YOLO()

path1 = os.listdir(path)

for i in path1:
    img = cv2.imread(path + i)

    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # convert into Image
    frame = Image.fromarray(np.uint8(frame))
    # detect
    tuxiang, zuobiao, label = yolo.detect_image(frame)
    print(zuobiao)

    frame = np.array(tuxiang)
    # RGBtoBGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.putText(frame, "fps= %.2f", (int((zuobiao[1] + zuobiao[3]) / 2), int((zuobiao[0] + zuobiao[2]) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("video", frame)

yolo.close_session()
