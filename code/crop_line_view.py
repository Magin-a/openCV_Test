#!/usr/bin/env python
#-*- coding:utf-8 -*-
import numpy as np
import cv2
import os

# input video
input_name = ''

# color indexing
red_color = (0, 0, 255)
green_color = (0, 255, 0)
blue_color = (255, 0, 0)

# Region of Interest
RoI = np.float32([[163, 130], [407, 130], [30, 200], [550, 200]])

# window size
x_size = 680
y_size = 480

# searching point for sliding window
search_point = [0, 0] #left_x, right_x
    
# calculating lane score of each grid
def grid_score(frame, left_high, right_low):
    score = np.sum(frame[left_high[1]:right_low[1], left_high[0]:right_low[0]])
    return score

# sliding window
def sliding_window(frame, search_point):
    frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    # grid number
    grid_x = 20
    grid_y = 10

    # grid number for searching
    left_grid = 3
    right_grid = 3

    # search starting point of next frame
    next_search = [0,0]

    # grid size
    margin_x = frame.shape[1] / grid_x   #34칸
    margin_y = frame.shape[0] / grid_y   #48칸

    # histogram of white pixel to get search starting point of left, right lane
    histogram = np.sum(frame[:,:], axis=0)

    # get midpoint of image and it become boundary of left and right lane
    midpoint = int(histogram.shape[0]/2)

    # search starting point from histogram
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # if there are search points from previous frame,
    # initialize starting point
    if search_point[0] != 0 and search_point[0] < midpoint:
        leftx_base = search_point[0]
    if search_point[1] != 0 and midpoint < search_point[1]: 
        rightx_base = search_point[1]
    
    # from leftx_base and rightx_base, get index of searching window
    leftx_current = int(leftx_base/margin_x)
    rightx_current = int(rightx_base/margin_x)

    # list of lane point
    left_line = []
    right_line = []

    #list of lane point x
    left_dot = []
    right_dot = []

    # start searching lane from bottom of image to top
    for grid in range(grid_y):
        # white pixels of each window are calculated to score
        left_score = []
        right_score = []

        # first, assume that there are no lane point
        left_point_exist = False
        right_point_exist = False


        # search white pixel of left side of image
        for left in range(left_grid):
            left_p1 = (int(margin_x * (leftx_current + left - int(left_grid / 2))), int(margin_y * (grid_y - grid - 1)))
            left_p2 = (int(margin_x * (leftx_current + left - int(left_grid / 2) + 1)), int(margin_y * (grid_y - grid)))
            left_score.append(grid_score(frame, left_p1, left_p2))      # calculate white pixel score of each window and append it to list
            cv2.rectangle(frame_color, left_p1, left_p2, red_color, 2)  # draw window
        
        # if there are no white pixel in left side
        if np.max(left_score) == 0:
            left_grid = 5   # set number of searching window of next frame to 5
        # if there are white pixel in left side
        else:
            leftx_current = leftx_current - int(left_grid / 2) + np.argmax(left_score)  # set searching point of upper window
            left_grid = 3                                                               # set number of searching window of next frame to 3
            left_point_exist = True


        # search white pixel of right side of image
        for right in range(right_grid):
            right_p1 = (int(margin_x * (rightx_current + right - int(right_grid / 2))), int(margin_y * (grid_y - grid - 1)))
            right_p2 = (int(margin_x * (rightx_current + right - int(right_grid / 2) + 1)), int(margin_y * (grid_y - grid)))
            right_score.append(grid_score(frame, right_p1, right_p2))         # calculate white pixel score of each window and append it to list
            cv2.rectangle(frame_color, right_p1, right_p2, blue_color, 2)   # draw window
        
        # if there are no white pixel in right side
        if np.max(right_score) == 0:
            right_grid = 5  # set number of searching window of next frame to 5
        # if there are white pixel in right side
        else:
            rightx_current = rightx_current - int(right_grid / 2) + np.argmax(right_score)    # set searching point of upper window
            right_grid = 3                                                                  # set number of searching window of next frame to 3
            right_point_exist = True
        
        # set left and right lane points
        left_point = (int(margin_x * leftx_current + margin_x / 2), int(margin_y * (grid_y - grid - 1) + margin_y / 2))
        right_point = (int(margin_x * rightx_current + margin_x / 2), int(margin_y * (grid_y - grid - 1) + margin_y / 2))

        # if right_point and left point are close each other, choice one point that have more points before
        if (right_point[0] - left_point[0]) < 200:
            if len(left_line) < len(right_line):
                left_point_exist = False
            elif len(left_line) > len(right_line):
                right_point_exist = False
    
        if left_point_exist == True:
            # draw left point
            cv2.line(frame_color, left_point, left_point, red_color, 10)
            if right_point_exist == True:
                # left point O, right point O
                cv2.line(frame_color, right_point, right_point, blue_color, 10) # draw right point
                # if calculated left point is in range
                if right_point[0] < x_size:
                    right_line.append(right_point)  # append it to list
                    right_dot.append(right_point[0])
            else:
                # left point O, right point X
                # assume that left lane is curved lane, and reinforce searching of left lane
                left_grid = 5
            # if calculated left point is in range
            if left_point[0] > 0:
                left_line.append(left_point)    # append it to list
                left_dot.append(left_point[0])
        else:
            if right_point_exist == True:
                # left point X, right point O
                # assume that right lane is curved lane, and reinforce searching of right lane
                right_grid = 5
                cv2.line(frame_color, right_point, right_point, blue_color, 10) # draw right point
                # if calculated right point is in range
                if right_point[0] < x_size:
                    right_line.append(right_point)  # append it to list
                    right_dot.append(right_point[0])
        
        # lane points of second window from bottom of image are saved to help next frame to set searching point
        if grid == 1:
            if left_point_exist == True:
                next_search[0] = left_point[0]
            if right_point_exist == True:
                next_search[1] = right_point[0]
        #print(len(left_dot), len(right_dot))

    return frame_color, left_line, right_line, next_search, left_dot, right_dot

def scharr_filter(frame):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # scharr x,y filtering for gradient detection
    img_scharr_x = cv2.Scharr(img_gray, cv2.CV_64F, 1, 0)
    img_scharr_x = cv2.convertScaleAbs(img_scharr_x)
    img_scharr_y = cv2.Scharr(img_gray, cv2.CV_64F, 0, 1)
    img_scharr_y = cv2.convertScaleAbs(img_scharr_y)

    # scharr x, y = scharr x + scharr y
    img_scharr = cv2.addWeighted(img_scharr_x, 1, img_scharr_y, 1, 0)

    _, white_line = cv2.threshold(img_scharr, 150, 255, cv2.THRESH_BINARY)
    return white_line

def top_view(frame):
    # set RoI
    pts = np.float32([[0, 0], [x_size, 0], [0, y_size], [x_size, y_size]])
    matrix = cv2.getPerspectiveTransform(RoI, pts)
    matrix_inv = cv2.getPerspectiveTransform(pts, RoI)
    frame = cv2.warpPerspective(frame, matrix_inv, (x_size, y_size))
    return frame

def bird_eye_view(frame):
    # set ROI
    pts = np.float32([[0, 0], [x_size, 0], [0, y_size], [x_size, y_size]])
    matrix = cv2.getPerspectiveTransform(RoI, pts)
    matrix_inv = cv2.getPerspectiveTransform(pts, RoI)
    frame = cv2.warpPerspective(frame, matrix, (x_size, y_size))
    return frame


def yellow_and_white_filter(image):
    """
    Filter the image to include only yellow and white pixels
    """
    # Filter white pixels
    white_threshold = 180 #130
    lower_white = np.array([white_threshold, white_threshold, white_threshold])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(image, lower_white, upper_white)
    white_image = cv2.bitwise_and(image, image, mask=white_mask)

    # Filter yellow pixels
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10, 40, 100])
    upper_yellow = np.array([23, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)
    
    # Combine the two above images
    out = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)
    
    return white_image, yellow_image, out


def roi(image):
    x = int(image.shape[1])
    y = int(image.shape[0])

    # 한 붓 그리기
    _shape = np.array(
        # [[int(0.1*x), int(y)], [int(0.1*x), int(0.1*y)], [int(0.4*x), int(0.1*y)], [int(0.4*x), int(y)], [int(0.7*x), int(y)], [int(0.7*x), int(0.1*y)],[int(0.9*x), int(0.1*y)], [int(0.9*x), int(y)], [int(0.2*x), int(y)]])
        [[120, 480], [120, 0], [220, 0], [220, 480], [460, 480], [460, 0], [560, 0], [560, 480], [120, 480]])
    mask = np.zeros_like(image)

    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, np.int32([_shape]), ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

def cal_steering(left_dot, right_dot):
    left_state, right_state = False, False
    steering_conversion = 1
    mid_avg = 0
    steer = 0

    if len(left_dot) != 0:
        left_state = True

    if len(right_dot) != 0:
        right_state = True

    if left_state and right_state:
        cal_count = min(len(left_dot), len(right_dot))
        
        for i in range(cal_count):
            mid_avg += (left_dot[i] + right_dot[i])/2
        steer = ((mid_avg/cal_count) - 340) * steering_conversion
    

    return steer

            
if __name__=='__main__':
    cap = cv2.VideoCapture(1) # 웹캠 연결
    # cap = cv2.VideoCapture(input_name)

    while(cap.isOpened()):
        # step0. Read frame
        ret, frame = cap.read()

        # step1. Resize image (Calibration skip)
        frame = cv2.resize(frame, (x_size, y_size))
        #cv2.imshow("1.resize image", frame)

        # step2. Bird-eye-view (Perspective transformation) 
        transformed_img = bird_eye_view(frame)
        #cv2.imshow('2. bird eye view image', transformed_img)

        w_f_r_img = roi(transformed_img)
        cv2.imshow("roi", w_f_r_img)

        # step3-1. Scharr filtering ====
        scharr_filtered_img = scharr_filter(w_f_r_img)
        cv2.imshow('3-1. Scharr filtered image', scharr_filtered_img)

        

        # step3-2. Yellow and White color filtering ====
        white_filtered, yellow_filtered, color_filtered_img = yellow_and_white_filter(w_f_r_img)
        color_filtered_img = cv2.cvtColor(color_filtered_img, cv2.COLOR_BGR2GRAY)
        _, color_filtered_img = cv2.threshold(color_filtered_img, 1, 255, cv2.THRESH_BINARY)
        # cv2.imshow('3-2. white_filtered image', white_filtered)
        # cv2.imshow('3-2. yellow_filtered image', yellow_filtered)
        # cv2.imshow('3-2. Yellow and White filtered image', color_filtered_img)

        # setp3-3. thickening detected lane
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thickened_color_filtered_img = cv2.dilate(color_filtered_img, kernel)
        #cv2.imshow('3-3. thickened', thickened_color_filtered_img)

        # step3-4. Final Filtering
        filtered_img = cv2.bitwise_and(scharr_filtered_img, thickened_color_filtered_img)
        #cv2.imshow('3-4. Filtered image', filtered_img)

        # step3-5. Median blur
        median_img=cv2.medianBlur(filtered_img, 5)

        # step4. Sliding Window
        window_searched_img, left_ptr, right_ptr, search_point, left_dot, right_dot = sliding_window(median_img, search_point)
        #cv2.imshow("sliding window", window_searched_img)
        steer_data = cal_steering(left_dot, right_dot)
        print(steer_data)


        # step5. Reverse perspective transform
        original = top_view(window_searched_img)
        #cv2.imshow('5. Reverse Perspective Transform', original)

        key = cv2.waitKey(30)
        if key == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()