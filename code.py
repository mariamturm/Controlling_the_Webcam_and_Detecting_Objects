import time
import cv2

first_frame = None
#frame_count = 0

video = cv2.VideoCapture(1)
time.sleep(0.07)

#a = 0
while True:
    #a = a+1
    check, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)


    if first_frame is None:
        first_frame = gray #first frame will be stored into gray and turned into grayscale
        # frame_count += 1
        continue #from second loop when first_frame is not none anymore this continue will make loop go from the top of the loop, it will not continue after if
    
    delta_frame = cv2.absdiff(first_frame,gray)

    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)

    (cnts,_) = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)


    # thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    # thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)


    

    cv2.imshow("Gray Frame", gray)
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Threshhold Frame", thresh_frame)
    cv2.imshow("Color Frame", frame)


    # frame_count += 1
    # if frame_count % 30 == 0:
    #     first_frame = gray

    key = cv2.waitKey(1)
    print(gray)
    print(delta_frame)

    if key == ord('q'):
        break
    


#print(f'Number of frames {a}')
video.release()
cv2.destroyAllWindows()