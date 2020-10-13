import cv2
import numpy as np
import picamera
from easygopigo3 import EasyGoPiGo3
import time

gpg = EasyGoPiGo3()

def canny(image):
    #this turns the image grey
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #this blurs the image, making edge detection more reliable
    blur = cv2.GaussianBlur(gray, (5,5),0)
    #this derives the array and thereby detects the change in
    #intensity in nearby pixles
    canny = cv2.Canny(blur,50,150)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image,(x1, y1), (x2, y2), (255,0,0),10)
            print((x1, y1),(x2, y2))
    return line_image

def waypoint_detection(image):
    NUM_IGNORED_ROWS = 60
    NUM_WAYPOINTS = 1

    num_rows = len(image)
    num_relevant_rows = (num_rows - NUM_IGNORED_ROWS)


    step = num_relevant_rows // (NUM_WAYPOINTS + 1)
    start = len(image[0])-(NUM_IGNORED_ROWS + step)
    waypoints = []
    for i in range(start, 0, -step):
        for j in range(len(image[i]) - 1, -1, -1):
            if image[i][j] == 255:
                waypoints.append((i,j))
                break
    if(waypoints == []):
        waypoints = [(0,len(image[0]))]
    return waypoints


def moveGoPiGo(waypoint):
    if ((waypoint[0][1] < (len(image[0])/2)+10) and (waypoint[0][1] > (len(image[0])/2)-10)):
        gpg.forward()
        time.sleep(0.1)
        gpg.stop()

    print("FUCK YOU")
    print(int(len(image[0])/2))
    print(waypoint[0][1] < (int(len(image[0])/2)))
    print(waypoint[0][1] > (int(len(image[0])/2)))
    if (waypoint[0][1] < (int(len(image[0])/2))):
        print("inside if 1")
        gpg.left()
        
        time.sleep(0.05)
        gpg.stop()
        print("bottom of if 1")
    if (waypoint[0][1] > (int(len(image[0])/2))):
        print("inside if 2")
        gpg.right()
       
        time.sleep(0.05)
        gpg.stop()

    
        

def regionOfInterest(image):
    amountTaken = 60
    rectangle = np.array([
        [(0,imageHeight), (0, amountTaken), (imageWidth,amountTaken), (imageWidth, imageHeight)]
        ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, rectangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


imageHeight = 384
imageWidth = 512
with picamera.PiCamera() as camera:
    camera.resolution = (imageWidth,imageHeight)
    camera.framerate = 30
    image = np.empty((imageHeight * imageWidth * 3), dtype=np.uint8)
    image = image.reshape((imageHeight,imageWidth,3))
    for frame in camera.capture_continuous(image, format="bgr", use_video_port=True):
        lane_image = np.copy(image)
        canny_image = canny(lane_image)
        croppedImage = regionOfInterest(canny_image)
        waypoints = waypoint_detection(croppedImage)
        
        moveGoPiGo(waypoints)
#        lines = cv2.HoughLinesP(croppedImage, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=10)
#        #print(lines)
#        line_image = display_lines(lane_image, lines)
#        combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
        cv2.imshow("lineVision",canny_image)
        cv2.waitKey(1)
