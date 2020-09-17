import cv2
# Basic drawing
# import numpy
#
cv2.namedWindow("Image") # Create window
# Capture camera video image
cap = cv2.VideoCapture(2)  # Create built-in camera variables

while(cap.isOpened()):  # isOpened()  Check if the camera is on
    ret,img = cap.read()  # Save the image information obtained by the camera as an img variable
    if ret == True:       # If the camera reads the image successfully
        cv2.imshow('Image',img)
        k = cv2.waitKey(100)
        if k == ord('a') or k == ord('A'):
            cv2.imwrite('test.jpg',img)
            break
cap.release()  # Turn off the camera
cv2.waitKey(0)