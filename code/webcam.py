import cv2 as cv

#영상이나 웹캡을 캡쳐하기 위한 VideoCapture
web = cv.VideoCapture(1) #인자는 포트번호 

if not web.isOpened():#캠 연결상태 확인
    print("Not link cam")
    exit()

while web.isOpened():
    status, frame = web.read() #연결상태 T/F, 캡쳐 이미지
    
    if status:
        target_img = frame.copy()
        img1_gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        rtn, img1_thr = cv.threshold(img1_gray, 127, 255, cv.THRESH_BINARY)
        contours, hierarchty = cv.findContours(img1_thr, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

        COLOR = (0, 200, 0)
        cv.drawContours(target_img, contours, -1, COLOR, 2)
        cv.imshow('test1', frame)
        cv.imshow('test2', target_img)
        cv.imshow("test3", img1_gray)


    if cv.waitKey(1) & 0xFF == ord('q'): #'q'로 종료
        break

web.release()
cv.destroyAllWindows()