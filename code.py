from datetime import datetime
import time
import cv2
import pandas

first_frame = None # Will store the first frame (used as background)

# status_list keeps track of motion status in consecutive frames.
# It starts with [None, None] so that we can safely compare previous and current states
# right from the beginning without getting an "index out of range" error.
# (Later in the code, status_list[-1] = current status, status_list[-2] = previous status) this will give "index out of range " error so none and none is used to avoid this
status_list = [None, None]

times = []  # Stores motion start and end times

df = pandas.DataFrame(columns = ["Start", "End"]) # Create pandas empty DataFrame with Start and End columns

# Start capturing video from webcam (1 means external camera, use 0 for built-in) in my case 0 turn on phoen camera because it is connected to computer and 1 uses computer camera
video = cv2.VideoCapture(1)

time.sleep(0.07) # I added this because without it when camera turns on it shows that whole visual in camera is object
#the camera needs a very short time to:
#adjust exposure and focus,
#start sending frames, and
#stabilize the image.

#infinite loop to read video frames continuously
while True:
    check, frame = video.read() #Read current frame; check=True if successful
    status = 0 #0 = no motion detected for this frame, so in the start frame there is no motion = no object

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)# Converts the color image to grayscale — makes motion detection faster & simpler (no color info).
    gray = cv2.GaussianBlur(gray,(21,21),0) # Apply Gaussian blur to remove noise and smooth the image

    # Set the first frame as reference (background)
    if first_frame is None:
        first_frame = gray #first frame will be stored into gray and turned into grayscale
        continue  # Skip the rest of the loop until we have a background
    
    #delta Shows the difference between the first (background) frame and 
    #the current frame — highlights what changed.
    # Compute difference between current frame and first frame
    delta_frame = cv2.absdiff(first_frame,gray) 

    #Turns that difference into pure black & white: white = movement, 
    #black = still. Used for contour detection.
    # Apply threshold to highlight areas with significant changes
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2) # Dilate (expand) the threshold image to fill in small gaps

    (cnts,_) = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find contours (outlines) of moving objects

    for contour in cnts: # Loop through all detected contours
        if cv2.contourArea(contour) < 70000: # Ignore small movements (noise) by checking contour area
            continue
        status = 1 # Motion detected!
        (x, y, w, h) = cv2.boundingRect(contour) # Get bounding box coordinates for the moving object
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)  # Draw a green rectangle around the detected object
    
    status_list.append(status)# Add current motion status (0 or 1) to the list
    
    # Detect transition: no motion -> motion (object appeared)
    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    
    # Detect transition: motion -> no motion (object disappeared)
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    # Show all processing frames in real time
    cv2.imshow("Gray Frame", gray)
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Threshhold Frame", thresh_frame)
    cv2.imshow("Color Frame", frame)

    # Wait 1 millisecond for key press
    key = cv2.waitKey(1)

    if key == ord('q'):# If user presses 'q', quit program
        if status == 1:# If motion is ongoing when exiting video, record final end time so that final object has start time and end time
            times.append(datetime.now())
        break
        
print(status_list)
print(times)

# Build list of motion time intervals from recorded timestamps
rows = []
for i in range(0, len(times), 2):
    rows.append({"Start": times[i], "End": times[i+1]})

# Convert list of dictionaries to a DataFrame
df = pandas.DataFrame(rows)
df.to_csv("Times.csv", index = False)# Save results to CSV file
df.to_json("Times.json", index = False)# Also save to JSON format     


# Release the camera and close all OpenCV windows
video.release()
cv2.destroyAllWindows()