import cv2
import numpy as np
import picamera
import math
from easygopigo3 import EasyGoPiGo3
GPG = EasyGoPiGo3()

def canny(image):
    #this turns the image grey
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #this blurs the image, making edge detection more reliable
    blur = cv2.GaussianBlur(gray, (5,5),0)
    #this derives the array and thereby detects the change in
    #intensity in nearby pixles
    canny = cv2.Canny(blur,50,150)
    return canny

def regionOfInterest(image):
    amountTaken = 60
    rectangle = np.array([
        [(0,imageHeight), (0, amountTaken), (imageWidth,amountTaken), (imageWidth, imageHeight)]
        ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, rectangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def CropImageFromTop(image, amount_px):
    cropped_image = image[amount_px:imageHeight, 0:imageWidth]
    return cropped_image


#Camera values
imageHeight = 480
imageWidth = 640



with picamera.PiCamera() as camera:
    camera.resolution = (imageWidth,imageHeight)
    camera.framerate = 30
    image = np.empty((imageHeight * imageWidth * 3), dtype=np.uint8)
    image = image.reshape((imageHeight,imageWidth,3))
    for frame in camera.capture_continuous(image, format="bgr", use_video_port=True):
        lane_image = np.copy(image)
        print(lane_image.shape)
        cropped_image = CropImageFromTop(lane_image, 60)
        #canny_image = canny(lane_image)
        #print(waypoints)
        #print(lines)
        cv2.imshow("lineVision",cropped_image)
        cv2.waitKey(1)


























