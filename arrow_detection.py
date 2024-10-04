import cv2
import numpy as np

def nothing(x):
    pass

#### SET TRACKBARS FOR CORNERS TO BE DETECTED 
cv2.namedWindow("img")
cv2.createTrackbar("quality", "img", 1, 100, nothing)

#### READ THE VIDEO
#url = "http://192.168.29.43:8080/video"
cap = cv2.VideoCapture(0)

#### CHECK IF THE VIDEO IS OPENED SUCCESSFULLY
if not cap.isOpened():
    print("Error: Video not loaded. Please check the file path.")
else:
    while True:
        #### READ A FRAME FROM THE VIDEO
        ret, frame = cap.read()
        
        #### CHECK IF THE FRAME IS READ PROPERLY
        if not ret:
            break

        #### EDGE DETECTION    
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Define range of red color in HSV
        lower_black = np.array([0, 70, 50])
        upper_black = np.array([10, 255, 255])
        # Threshold the HSV image to get only black colors
        mask = cv2.inRange(hsv, lower_black, upper_black)

        #### FIND THE CONTOURS OF THE BLACK REGIONS
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #### DRAW RECTANGLES AROUND THE BLACK REGIONS
        for contour in contours:
            # Get the coordinates of the bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            # Draw rectangles around the black regions
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            frame = cv2.putText(frame, 'OpenCV', ((200,320)),  cv2.FONT_HERSHEY_SIMPLEX ,  
                       2, (255,0,255), 5, cv2.LINE_AA)
            print("coordinates", x,y,x+w,y+h)

        #### SHOW THE FRAME
        cv2.imshow('Black Regions', frame)

        #### CHECK FOR KEYBOARD INTERRUPT (PRESS 'q' TO EXIT)
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    #### RELEASE THE VIDEO CAPTURE OBJECT AND CLOSE WINDOWS
    cap.release()
    cv2.destroyAllWindows()