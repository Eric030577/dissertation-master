import pyrealsense2 as rs
import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
# -------------------------------------#
#      Call camera to detect
# -------------------------------------#


import time
from keras.layers import Input
from yolo import YOLO
from PIL import Image

import time
import os

# Communication part
import socket

server = socket.socket()  # AF_INET、SOCK_STREAM
server.bind(('192.168.31.238', 6868))  # Bind the host number and port to the socket
server.listen()  # Set up and start the TCP listener
# Load YOLO class
yolo = YOLO()
# Load depth camera class
pipeline = rs.pipeline()

cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Set the mode of alignment (here is depth alignment color,
# thus keep color map unchanged, transform depth map)
align_to = rs.stream.color
#  align_to = rs.stream.depth

alignedFs = rs.align(align_to)
print("YOLO loading is completed")
profile = pipeline.start(cfg)

# Establish communication
conn, addr = server.accept()
# Passively accepts a TCP connection and waits for the connection to arrive
data = conn.recv(1024)

while True:
    fs = pipeline.wait_for_frames()
    aligned_frames = alignedFs.process(fs)

    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not depth_frame or not color_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    depth_image = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.1), cv.COLORMAP_JET)

    color_image = cv.cvtColor(color_image, cv.COLOR_BGR2RGB)
    # convert to Image
    frame = Image.fromarray(np.uint8(color_image))
    # start to detect
    tuxiang, zuobiao, label = yolo.detect_image(frame)

    change = zuobiao
    if len(zuobiao) != 0:
        if zuobiao[0] == zuobiao[1] == zuobiao[2] == zuobiao[3] == 0:
            pass
        else:
            # obtain object depth coordinate
            dist_to_center = depth_frame.get_distance(int((change[1] + change[3]) / 2),
                                                      int((change[0] + change[2]) / 2))

            zuobiao = (str(zuobiao[0]) + "#" + str(zuobiao[1]) + "#" + str(zuobiao[2]) + "#" + str(
                zuobiao[3]) + "#" + label + "#" + str(dist_to_center)).encode("utf-8")

            conn.send(zuobiao)  # Converts the received data to uppercase and sends it back
    frame = np.array(tuxiang)
    # RGBtoBGR Meet the openCV display format
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    print("zuobiao", (int((change[1] + change[3]) / 2), int((change[0] + change[2]) / 2)))
    frame = cv.putText(frame, "o", (int((change[1] + change[3]) / 2), int((change[0] + change[2]) / 2)),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.imshow("video", frame)

    cv.waitKey(1)
server.close()

yolo.close_session()
