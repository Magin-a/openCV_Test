import numpy as np
import cv2 as cv

def region_of_interest(img, vertices, color3=(255, 255, 255),color1 = 255):
    mask = np.zeros_like(img) #같은 사이즈의 빈 이미지

    if len(img.shape) > 2:
        color = color3

    else:
        color = color1

    #vertices로 정한 점들로 이뤄진 다각형부분(ROI 설정 부분)을 color로 채움
    cv.fillPoly(mask, vertices, color)

    ROI_image = cv.bitwise_and(img, mask)
    return ROI_image

def mark_img(img, blue_treshold=200, green_threshold=200, red_threshold=200):
    bgr_threshold = [blue_treshold, green_threshold, red_threshold]

    thresholds = (image[:,:,0] < bgr_threshold[0]) \
                | (image[:,:,1] < bgr_threshold[1]) \
                | (image[:,:,2] < bgr_threshold[2])
    mark[thresholds] = [0,0,0]
    return mark

cap = cv.VideoCapture("openCV_Test\code\IMAGE\testvideo.mp4")

while(cap.isOpened()):
    ret, image = cap.read()
    height, width = image.shape[:2] # 이미지 높이, 너비

    # 사다리꼴 모형의 Points
    vertices = np.array([[(50,height),(width/2-45, height/2+60), (width/2+45, height/2+60), (width-50,height)]], dtype=np.int32)
    roi_img = region_of_interest(image, vertices, (0,0,255)) # vertices에 정한 점들 기준으로 ROI 이미지 생성

    mark = np.copy(roi_img) # roi_img 복사
    mark = mark_img(roi_img) # 흰색 차선 찾기

    # 흰색 차선 검출한 부분을 원본 image에 overlap 하기
    color_thresholds = (mark[:,:,0] == 0) & (mark[:,:,1] == 0) & (mark[:,:,2] > 200)
    image[color_thresholds] = [0,0,255]

    cv.imshow('results',image) # 이미지 출력
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# Release
cap.release()
cv.destroyAllWindows()