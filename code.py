from datetime import datetime
import time
import cv2
import pandas

first_frame = None
#frame_count = 0
status_list = [None, None]
times = []

df = pandas.DataFrame(columns = ["Start", "End"])

video = cv2.VideoCapture(1)
time.sleep(0.07)

#a = 0
while True:
    #a = a+1
    check, frame = video.read()
    status = 0

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
        if cv2.contourArea(contour) < 70000:
            continue
        status = 1
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    
    status_list.append(status)
    
    
    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())



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
    #print(gray)
    #print(delta_frame)

    if key == ord('q'):
        if status == 1:
            times.append(datetime.now())
        break
    
    #print(status)
    
print(status_list)
print(times)

rows = []
for i in range(0, len(times), 2):
    rows.append({"Start": times[i], "End": times[i+1]})

df = pandas.DataFrame(rows)
df.to_csv("Times.csv", index = False)
df.to_json("Times.json", index = False)     

   
#print(f'Number of frames {a}')
video.release()
cv2.destroyAllWindows()