import cv2
import numpy as np
import picamera
# import math
from easygopigo3 import EasyGoPiGo3

gpg = EasyGoPiGo3()


def cropImage(image, top, bottom, left, right):
    newImage = image[top:bottom, left:right]
    return newImage


def findContours(image):
    newImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurredImage = cv2.GaussianBlur(newImage, (1, 1), 0)
    _, img_bw = cv2.threshold(blurredImage, 75, 255, cv2.THRESH_BINARY_INV)
    im, contours, hierarchy = cv2.findContours(img_bw, 1, cv2.CHAIN_APPROX_NONE)
    return img_bw, contours


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
        # gpg.steer(100 - power_proportion, 100 - power_proportion)
        if cx < width/3:
            gpg.right()
        #if width/3 < cx < width*2/3:
           # gpg.forward()
        if cx < width*2/3:
            gpg.left()
    else:
        gpg.set_speed(0)
        # if cx >= 340:
        #     gpg.left()
        # if 340 > cx > 300:
        #     gpg.forward()
        # if cx < 300:
        #     gpg.right()
        # else:
        #     print("Can't see the line")


imageHeight = 480
imageWidth = 640

with picamera.PiCamera() as camera:
    camera.resolution = (imageWidth, imageHeight)
    camera.framerate = 30
    image = np.empty((imageHeight * imageWidth * 3), dtype=np.uint8)
    image = image.reshape((imageHeight, imageWidth, 3))
    for frame in camera.capture_continuous(image, format="bgr", use_video_port=True):
        gpg.set_speed(0)
        lane_image = np.copy(image)
        croppedImage = cropImage(lane_image, 200, 480, 40, 600)
        resized_image = cv2.resize(croppedImage, (240, 100))
        thresh, contours = findContours(resized_image)
        gpg.set_speed(10)
        motorControl(thresh, resized_image, contours)
        cv2.imshow("lineVision", resized_image)
        cv2.imshow("lineVision1", thresh)

        cv2.waitKey(1)
