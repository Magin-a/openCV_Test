import cv2 as cv
import numpy as np

img = "openCV_Test\code\IMAGE\expressway.jpg"

img1 = cv.imread(img)  #ex) cv.imread(r'C:\Users\starry_night.jpg')
target_img = img1.copy()
img1_gray = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
rtn, img1_thr = cv.threshold(img1_gray, 127, 255, cv.THRESH_BINARY)

contours, hierarchty = cv.findContours(img1_thr, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

COLOR = (0, 200, 0)
cv.drawContours(target_img, contours, -1, COLOR, 2)


#cv.imshow('img', img1)
#cv.imshow("gray", img1_gray)
cv.imshow('contours', target_img)
cv.waitKey(0)
cv.destroyAllWindows()