import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#영상이나 웹캡을 캡쳐하기 위한 VideoCapture
web = cv.VideoCapture(1) #인자는 포트번호
status, frame = web.read()# 웹캠이나 영상일 경우 frame을 추출해서 사용

if not web.isOpened():#캠 연결상태 확인
    print("Not link cam")
    exit()

# def trans_Gray(img):
#     return cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# convert_Gray = trans_Gray(frame)

# def gaussian_blur(img, kernel_size):
#     return cv.GaussianBlur(img, (kernel_size, kernel_size), 0)

# kernel_size = 5
# blur_gray = gaussian_blur(convert_Gray, kernel_size)

# def canny(img, low_threshold, high_treshold):
#     return cv.Canny(img, low_threshold, high_treshold)

# edge = canny(blur_gray, 50, 200)
# plt.figure(figsize=(10, 8))
# plt.imshow(edge, cmap='edge')
# plt.show()

while web.isOpened():
    status, frame = web.read() #연결상태 T/F, 캡쳐 이미지
    
    if status:
        target_img = frame.copy()
        img1_gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        rtn, img1_thr = cv.threshold(img1_gray, 127, 255, cv.THRESH_BINARY)
        contours, hierarchty = cv.findContours(img1_thr, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

        COLOR = (0, 200, 0)
        # cv.drawContours(target_img, contours, -1, COLOR, 2)
        # cv.imshow('test1', frame)
        # cv.imshow('test2', target_img)
        cv.imshow("test3", img1_gray)

    if cv.waitKey(1) & 0xFF == ord('q'): #'q'로 종료
        break


web.release()
cv.destroyAllWindows()