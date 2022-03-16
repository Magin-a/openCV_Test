import cv2 as cv

#불러오기
img = cv.imread("openCV_Test\IMAGE\lulu.jpg")  #ex) cv.imread(r'C:\Users\starry_night.jpg')
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()
