import cv2
import numpy as np
import time
import picamera
import matplotlib.pyplot as plt

#this takes a snapshot with the camera, commented out while
#testing
imageHeight = 240
imageWidth = 320
with picamera.PiCamera() as camera:
    camera.resolution = (imageWidth,imageHeight)
    camera.framerate = 24
    time.sleep(2)
    image = np.empty((imageHeight * imageWidth * 3), dtype=np.uint8)
    camera.capture(image, 'bgr')
    image = image.reshape((240,320,3))

def canny(image):
    #this turns the image grey
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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

#this is just a test to see what the fuck is going on
#this is an array made from the image
#image = cv2.imread('image.jpg')
#this is a copy of the array above
lane_image = np.copy(image)

canny = canny(lane_image)

cv2.imshow("asscum",regionOfInterest(canny))
cv2.waitKey(0)