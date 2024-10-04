import cv2
import numpy as np

def nothing(x):
    pass

#### SET TRACKBARS FOR CORNERS TO BE DETECTED 
cv2.namedWindow("img")
cv2.createTrackbar("quality", "img", 1, 100, nothing)

#### READ THE IMAGE
img = cv2.imread(r"D:\Robomanthan\Arrow_detection\testleft.jpg")

#### CHECK IF THE IMAGE IS LOADED SUCCESSFULLY
if img is None:
    print("Error: Image not loaded. Please check the file path.")
else:
    #### EDGE DETECTION    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define range of black color in HSV
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])
    # Threshold the HSV image to get only black colors
    mask = cv2.inRange(hsv, lower_black, upper_black)

    #### FIND THE CONTOURS OF THE BLACK REGIONS
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #### PRINT THE COORDINATES OF THE BLACK REGIONS
    print("Coordinates of black regions:")
    for contour in contours:
        # Get the coordinates of the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        # Print the coordinates
        print("Top-left: ({}, {})".format(x, y))
        print("Bottom-right: ({}, {})".format(x + w, y + h))

    #### DRAW RECTANGLES AROUND THE BLACK REGIONS
    for contour in contours:
        # Get the coordinates of the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        # Draw rectangles around the black regions
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        img = cv2.putText(img, 'OpenCV', ((200,320)),  cv2.FONT_HERSHEY_SIMPLEX ,  
                   2, (255,0,255), 5, cv2.LINE_AA)
        print("coordinates", x,y,x+w,y+h)
    #### SHOWING THE OUTPUT
    cv2.imshow('Black Regions', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
