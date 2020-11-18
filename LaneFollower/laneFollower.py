import cv2
import numpy as np
import picamera
import math
import select
import sys
from easygopigo3 import EasyGoPiGo3

gpg = EasyGoPiGo3()


def stop_program():
    i, o, e = select.select([sys.stdin], [], [], 0.0001)
    for s in i:
        if s == sys.stdin:
            input = sys.stdin.readline(1)
            return True
    return False

def cropImage(image, top, bottom, left, right):
    newImage = image[top:bottom, left:right]
    return newImage


def findContours(image):
    newImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurredImage = cv2.GaussianBlur(newImage, (1, 1), 0)
    _, img_bw = cv2.threshold(blurredImage, 200, 255, cv2.THRESH_BINARY)
    im, contours, hierarchy = cv2.findContours(img_bw, 1, cv2.CHAIN_APPROX_NONE)
    return img_bw, contours

def turnRobot(x_coord, center_threshold, img_width):
    gpg.set_speed(0)
    center = img_width / 2
    left_threshold = center - center_threshold / 2
    right_threshold = center + center_threshold / 2

    if x_coord <= left_threshold:
        gpg.left()
    elif x_coord <= right_threshold:
        gpg.forward()
    else:
        gpg.right()

def control_robot(turn_rate):
    speed_percentage = 25
    FORWARD_POWER = 50

    left_power = FORWARD_POWER + turn_rate * speed_percentage
    right_power = FORWARD_POWER - turn_rate * speed_percentage
    # print("Left Power: {}, Right Power: {}".format(left_power,right_power))

    gpg.set_motor_power(gpg.MOTOR_LEFT, left_power)
    gpg.set_motor_power(gpg.MOTOR_RIGHT, right_power)

def otherTurnRobot(cx):
    gpg.set_speed(20)
    if cx >= 240 * 2 / 3:
        gpg.right()
    if 240 / 3 > cx > 240 * 2 / 3:
        gpg.forward()
    if cx < 240 / 3:
        gpg.left()
    else:
        print("Can't see the line")

def otherOtherTurnRobot(cx):
    gpg.set_speed(10)
    gpg.steer()

def motorControl(image, imageForDrawing, contours):
    if len(contours) > 0:


        contour_area = max(contours, key=cv2.contourArea)
        moment = cv2.moments(contour_area)

        width = len(image[0])
        height = len(image)

        if(moment['m00'] != 0):
            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])
        else:
            return

        cv2.drawContours(imageForDrawing, contours, -1, (0, 255, 0), 1)
        cv2.line(imageForDrawing, (cx-10, cy), (cx+10, cy), (0, 0, 255), 1)
        cv2.line(imageForDrawing, (cx, cy-10), (cx, cy+10), (0, 0, 255), 1)


        power_proportion = abs(cx)
        print(power_proportion)

        x0, y0 = newimgsize[0] / 2, newimgsize[1]
        katet =  x0 - cx
        motKat = y0 - cy
        hypotenus = math.sqrt(math.pow(math.fabs(katet), 2) + math.pow(math.fabs(motKat), 2))
        #angleRad = math.acos(katet / hypotenus)
        #angle = angleRad * 180 / math.pi
        cosine = katet/hypotenus
        #turnRobot(cx, 100, 240)
        control_robot(cosine)



        # gpg.steer(100 - power_proportion, 100 - power_proportion)
        # if cx < width/3:
        #     gpg.right()
        # #if width/3 < cx < width*2/3:
        #    # gpg.forward()
        # if cx < width*2/3:
        #     gpg.left()

imageHeight = 480
imageWidth = 640
newimgsize = (240, 100)
with picamera.PiCamera() as camera:
    camera.resolution = (imageWidth, imageHeight)
    camera.framerate = 30
    image = np.empty((imageHeight * imageWidth * 3), dtype=np.uint8)
    image = image.reshape((imageHeight, imageWidth, 3))
    for frame in camera.capture_continuous(image, format="bgr", use_video_port=True):
        if stop_program():
            gpg.set_motor_power(gpg.MOTOR_LEFT + gpg.MOTOR_RIGHT, 0)
            break
        print("----------------\n")

        lane_image = np.copy(image)
        croppedImage = cropImage(lane_image, 200, 480, 40, 600)
        resized_image = cv2.resize(croppedImage, newimgsize)
        thresh, contours = findContours(resized_image)
        gpg.set_speed(20)
        motorControl(thresh, resized_image, contours)
        cv2.imshow("lineVision", resized_image)
        cv2.imshow("lineVision1", thresh)

        cv2.waitKey(1)
