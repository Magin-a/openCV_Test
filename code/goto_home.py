import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#영상 불러오기
cam = cv.VideoCapture(1)# 포트번호

def cvt_Gray(img):
    return cv.cvtColor(img, cv.COLOR_RGB2GRAY)

def gaussian_blur(img, kernel_size):
    return cv.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_treshold):
    return cv.Canny(img, low_threshold, high_treshold)


if not cam.isOpened():
    print("Not connect cam")
    exit()

while cam.isOpened():
    state,  frame = cam.read()

    if state:
        target_img = frame.copy()
        img_gray = cvt_Gray(frame)
        rtn, img_thr = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)
        contours, hierarchty = cv.findContours(img_thr, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

        color = (0, 0, 255)
        cv.drawContours(target_img, contours, -1, color, 2)
        cv.imshow('test', target_img)

cam.release()
cv.destroyAllWindows()