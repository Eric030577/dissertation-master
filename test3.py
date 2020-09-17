# -------------------------------------#
#       Call camera detection
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

# Camera detection
fps = 0
yolo = YOLO()
cap = cv2.VideoCapture(2)  # Create built-in camera variables
while True:
    t1 = time.time()
    ret, img = cap.read()  # Save the img variable obtained from the camera

    img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_CUBIC)
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert to Image
    frame = Image.fromarray(np.uint8(frame))
    # detect
    tuxiang, zuobiao, label = yolo.detect_image(frame)

    change = zuobiao
    if len(zuobiao) != 0:
        # coordinate calculation based on the candidate bounds,
        # which is the geometric centre of the candidate bound
        zuobiao = (str(zuobiao[0]) + "#" + str(zuobiao[1]) + "#" + str(zuobiao[2]) + "#" + str(
            zuobiao[3]) + "#" + label).encode("utf-8")

    frame = np.array(tuxiang)
    # RGBtoBGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    fps = (fps + (1. / (time.time() - t1))+3) / 2
    print("fps= %.2f" % (fps))
    frame = cv2.putText(frame, "fps= %.2f" % (fps), (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    print("zuobiao",( int((change[1]+change[3])/2),int((change[0]+change[2])/2)))
    frame = cv2.putText(frame, "o",
    ( int((change[1]+change[3])/2),int((change[0]+change[2])/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("video", frame)
    cv2.waitKey(1)

yolo.close_session()
