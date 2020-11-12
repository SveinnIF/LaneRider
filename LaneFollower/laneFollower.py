import cv2
import numpy as np
import picamera
from easygopigo3 import EasyGoPiGo3

gpg = EasyGoPiGo3()


def cropImage(image, top, bottom, left, right):
    newImage = image[top:bottom, left:right]
    return newImage


def findContours(image):
    _, img_gs = cv2.threshold(image, 127, 255, cv2.COLOR_BGR2GRAY)
    blurredImage = cv2.GaussianBlur(img_gs, (5, 5), 0)
    _, img_bw = cv2.threshold(blurredImage, 127, 255, cv2.THRESH_BINARY)
    #print(img_bw)
    img_cnt, contours, _ = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return img_bw, contours

def motorControl(image, contours):
    if len(contours) > 0:
        contour_area = max(contours, key=cv2.contourArea)
        moment = cv2.moments(contour_area)

        cx = int(moment['m10']/moment['m00'])
        cy = int(moment['m01']/moment['m00'])

        cv2.line(image,(cx, 0), (cx,640), (255, 0, 0), 1)
        cv2.line(image,(0, cy), (420, cy), (255, 0, 0), 1)

        cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

        if cx >= 120:
            gpg.left()
        if 120 > cx > 50:
            gpg.forward()
        if cx < 50:
            gpg.right()
        else:
            print("Can't see the line")

imageHeight = 480
imageWidth = 640

with picamera.PiCamera() as camera:
    camera.resolution = (imageWidth,imageHeight)
    camera.framerate = 30
    image = np.empty((imageHeight * imageWidth * 3), dtype=np.uint8)
    image = image.reshape((imageHeight,imageWidth,3))
    for frame in camera.capture_continuous(image, format="bgr", use_video_port=True):
        lane_image = np.copy(image)
        croppedImage = cropImage(lane_image, 0, 480, 0, 640)
        cv2.imshow("lineVision", croppedImage)
        print(croppedImage)
        #thresh, contours = findContours(croppedImage)
        #motorControl(thresh, contours)
        cv2.imshow("lineVision1", thresh)
        cv2.waitKey(1)
