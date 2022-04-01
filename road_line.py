import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


#과정
#이미지 불러오기 -> 색변환(gray) -> 가우시안 필터 -> 캐니 엣지 디텍터


#이미지 불러오기 (matplotlib.pyplot)
img = cv.imread("IMAGE/expressway.jpg")
img_gray = cv.imread("IMAGE/expressway_gray.jpg")

plt.figure(figsize=(10, 8))
print(type(img), img.shape)

plt.imshow(img)
plt.show()


#차선 인식을 위한 색 변환 함수
def TransGray(img):
    return cv.cvtColor(img, cv.COLOR_RGB2GRAY)

gray = TransGray(img)
plt.figure(figsize=(10, 8))
plt.imshow(gray, cmap='gray')
plt.show()


#블러링
#- 블러링은 거친 느낌의 사진을 부드럽게 만들거나, 혹은 영산 인식 부분에서, 잡음의 영향을 제거하기 위한 기법
def gaussian_blur(img, kernel_size):
    return cv.GaussianBlur(img, (kernel_size, kernel_size), 0)

kernel_size = 5
blur_gray = gaussian_blur(gray, kernel_size)

plt.figure(figsize=(10, 8))
plt.imshow(blur_gray, cmap='gray')
plt.show()

def canny(img, low_threshold, high_treshold):
    return cv.Canny(img, low_threshold, high_treshold)


edge = canny(blur_gray, 50, 200)
# plt.figure(figure = (10, 8))
plt.imshow(edge, cmap='gray')
plt.show()

