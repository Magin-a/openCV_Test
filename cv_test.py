# import numpy as np
# import cv2 as cv

# img = cv.imread("C:\\Users\\박영웅\\Desktop\\python연습장\\이온2파이썬\\openCV\\사진\\루루.jpg")


# cv.imshow("cat", img)
# cv.waitKey()
# cv.destroyAllWindows()

import cv2 as cv


img = cv.imread("IMAGE\\lulu.jpg")  #ex) cv.imread(r'C:\Users\starry_night.jpg')
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()

