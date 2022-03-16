import cv2 as cv

#불러오기
img = cv.imread("파일 경로")  #ex) cv.imread(r'C:\Users\starry_night.jpg')
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()
