import numpy as np
import cv2 as cv

cap = cv.VideoCapture("openCV_Test\code\IMAGE\testvideo.mp4")

width = cap.get(cv.CAP_PROP_FRAME_WIDTH) 
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT) 

cap.set(cv.CAP_PROP_FRAME_WIDTH, 640) # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width/2)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480) # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height/2)
print(width) # 640
print(height)
while True:
    ret, img = cap.read()
    cv.imshow('test', img)

    if cv.waitKey(10) == ord('q'):
        break
# Release
cap.release()
