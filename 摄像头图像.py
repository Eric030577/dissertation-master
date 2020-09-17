
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
import cv2
# 基本绘图
# import numpy
#
cv2.namedWindow("Image") #创建窗口
#抓取摄像头视频图像
cap = cv2.VideoCapture(0)  #创建内置摄像头变量

while(cap.isOpened()):  #isOpened()  检测摄像头是否处于打开状态
    ret,img = cap.read()  #把摄像头获取的图像信息保存之img变量
    if ret == True:       #如果摄像头读取图像成功
        cv2.imshow('Image',img)
        k = cv2.waitKey(100)
        if k == ord('a') or k == ord('A'):
            cv2.imwrite('test.jpg',img)
            break
cap.release()  #关闭摄像头
cv2.waitKey(0)
cv2.destroyAllWindow()