import cv2
import numpy as np
import picamera

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
    return line_image

def waypoint_detection(image):
    NUM_IGNORED_ROWS = 60
    NUM_WAYPOINTS = 3

    num_rows = len(image.array)
    num_relevant_rows = (num_rows - NUM_IGNORED_ROWS)

    start = NUM_IGNORED_ROWS
    step = num_relevant_rows // (NUM_WAYPOINTS + 1)

    for i in range(start, num_rows, step):
        for j in range(len(image.array[i]),0):
            if image.array[i][j] is not None:
                waypoints.append([i][j])
                break


def regionOfInterest(image):
    amountTaken = 60
    rectangle = np.array([
        [(0,imageHeight), (0, amountTaken), (imageWidth,amountTaken), (imageWidth, imageHeight)]
        ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, rectangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

imageHeight = 768
imageWidth = 1024
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
        print(waypoints)
        #lines = cv2.HoughLinesP(croppedImage, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=10)
        #print(lines)
        #line_image = display_lines(lane_image, lines)
        #combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
        cv2.imshow("asscum",croppedImage)
        cv2.waitKey(1)
