import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


#과정
#이미지 불러오기 -> 색변환(gray) -> 가우시안 필터 -> 캐니 엣지 디텍터


#이미지 불러오기 (matplotlib.pyplot)
img = cv.imread("openCV_Test\code\IMAGE\solidWhiteCurve.jpg")
# img_gray = cv.imread("code\IMAGE/expressway_gray.jpg")

# plt.figure(figsize=(10, 8))
print(type(img))
#print(img.shape) #shape(높이, 너비, channel)

# plt.imshow(img)
# plt.show()


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

kernel_size = 5 #연산을 수행할 때 윈도우의 크기
blur_gray = gaussian_blur(gray, kernel_size)

plt.figure(figsize=(10, 8))
plt.imshow(blur_gray, cmap='gray')
plt.show()

def canny(img, low_threshold, high_treshold):
    return cv.Canny(img, low_threshold, high_treshold)


edge = canny(blur_gray, 50, 200)
plt.figure(figsize = (10, 8))
plt.imshow(edge, cmap='gray')
plt.show()

#공백사진
mask = np.zeros_like(img)
# plt.figure(figsize=(10, 8))
# plt.imshow(mask, cmap='gray')


#카메라 인식할 영역 표시
if len(img.shape) > 2:
    channel_count = img.shape[2]
    ignore_mask_color = (255,) * channel_count

else:
    ignore_mask_color = 255

imshape = img.shape #(750, 1000, 3)



#선택한 영역안에 있는 차선 인식하기
def region_of_interest(img, vertices):
    #공백 마스크 설정
    mask = np.zeros_like(img)

    if len(img.shape) >2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count

    else:
        ignore_mask_color = 255

    cv.fillPoly(mask, vertices, ignore_mask_color)

    #마스크 픽셀이 0이 아닌 경우 이미지 반환
    masked_image = cv.bitwise_and(img, mask)
    return masked_image


#선긋기전 좌표점
vertices = np.array([[(100, imshape[0]),(450, 320), (550, 320),(imshape[1]-20, imshape[0])]], dtype=np.int32)
# vertices = np.array([[(0, 2650), (4000, 2650), (2500, 1050), (1550, 1050)]], dtype=np.int32) #
# cv.fillPoly(mask, vertices, ignore_mask_color)
mask = region_of_interest(edge, vertices)

# plt.figure(figsize=(10, 8))
# plt.imshow(mask, cmap='gray')
# plt.show()


#선그리기 함수1
def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(img, (x1,y1), (x2, y2), color, thickness)

#선그리기 함수2
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv.HoughLinesP(img, rho, theta, threshold, np.array([]),
        minLineLength=min_line_len,
        maxLineGap = max_line_gap)
    
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

rho = 2 #
theta = np.pi/180 # 
threshold = 90 #
min_line_len = 120 #
max_line_gap = 150 #

lines = hough_lines(mask, rho, theta, threshold, min_line_len, max_line_gap)

plt.figure(figsize=(10, 8))
plt.imshow(lines, cmap='gray')
plt.show()


#원본 img와 합치기
def weighted_img(img, initial_img, a=0.8, b=1, c=0):
    return cv.addWeighted(initial_img, a, img ,b, c)

lines_edge = weighted_img(lines, img)

plt.figure(figsize=(10, 8))
plt.imshow(lines_edge)
plt.show()