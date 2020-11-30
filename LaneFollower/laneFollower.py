import cv2
import numpy as np
import picamera
import math
import select
import sys
from easygopigo3 import EasyGoPiGo3

gpg = EasyGoPiGo3()

# values for configuration
imageHeight = 480
imageWidth = 640
newimgsize = (240, 100)
speed_percentage = 25
FORWARD_POWER = 50


def stop_program():
    i, o, e = select.select([sys.stdin], [], [], 0.0001)
    for s in i:
        if s == sys.stdin:
            input = sys.stdin.readline(1)
            return True
    return False


def cropImage(image, top, bottom, left, right):
    newImage = image[top:bottom, left:right]
    cv2.resize(newImage, newimgsize)
    return newImage


def findContours(image):
    newImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurredImage = cv2.GaussianBlur(newImage, (1, 1), 0)
    _, img_bw = cv2.threshold(blurredImage, 200, 255, cv2.THRESH_BINARY)
    im, contours, hierarchy = cv2.findContours(img_bw, 1, cv2.CHAIN_APPROX_NONE)
    return img_bw, contours


def motion_control_algorithm(cx, cy):

    x0, y0 = newimgsize[0] / 2, newimgsize[1]
    katet = x0 - cx
    motKat = y0 - cy
    hypotenus = math.sqrt(math.pow(math.fabs(katet), 2) + math.pow(math.fabs(motKat), 2))
    cosine = katet / hypotenus
    turn_rate = -cosine

    left_power = FORWARD_POWER + turn_rate * speed_percentage
    right_power = FORWARD_POWER - turn_rate * speed_percentage

    gpg.set_motor_power(gpg.MOTOR_LEFT, left_power)
    gpg.set_motor_power(gpg.MOTOR_RIGHT, right_power)


def point_detection(imageForDrawing, contours):
    if len(contours) > 0:


        contour_area = max(contours, key=cv2.contourArea)
        moment = cv2.moments(contour_area)

        if(moment['m00'] != 0):
            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])
        else:
            return

        cv2.drawContours(imageForDrawing, contours, -1, (0, 255, 0), 1)
        cv2.line(imageForDrawing, (cx-10, cy), (cx+10, cy), (0, 0, 255), 1)
        cv2.line(imageForDrawing, (cx, cy-10), (cx, cy+10), (0, 0, 255), 1)

        return cx, cy


with picamera.PiCamera() as camera:
    camera.resolution = (imageWidth, imageHeight)
    camera.framerate = 30
    image = np.empty((imageHeight * imageWidth * 3), dtype=np.uint8)
    image = image.reshape((imageHeight, imageWidth, 3))
    for frame in camera.capture_continuous(image, format="bgr", use_video_port=True):

        #this code stops the program if the enter key is ever pressed
        if stop_program():
            gpg.set_motor_power(gpg.MOTOR_LEFT + gpg.MOTOR_RIGHT, 0)
            break

        lane_image = np.copy(image)
        resized_image = cropImage(lane_image, 200, 480, 40, 600)
        thresh, contours = findContours(resized_image)
        #  the * unpacks the two values from point detection into two comma separated values
        motion_control_algorithm(*point_detection(resized_image, contours))

        cv2.imshow("camera feed", resized_image)
        cv2.imshow("threshold feed", thresh)

        cv2.waitKey(1)
