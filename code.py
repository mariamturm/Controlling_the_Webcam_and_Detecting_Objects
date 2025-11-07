import time
import cv2

video = cv2.VideoCapture(1)

a = 0
while True:
    a = a+1
    check, frame = video.read()

    cv2.imshow("Capturing", frame)

    key = cv2.waitKey(1)
    
    print(frame)

    if key == ord('q'):
        break

print(f'Number of frames {a}')
video.release()
cv2.destroyAllWindows()