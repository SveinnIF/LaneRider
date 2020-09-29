import cv2
import numpy as np
import time
import picamera
import easygopigo3
import easysensors
from easygopigo3 import EasyGoPiGo3
import matplotlib.pyplot as plt
# import the time library for the sleep function
import time
# import the GoPiGo3 drivers
from easygopigo3 import EasyGoPiGo3

# Create an instance of the GoPiGo3 class.
# GPG will be the GoPiGo3 object.
gpg = EasyGoPiGo3()

print("Move the motors forward freely for 1 second.")
gpg.forward()
time.sleep(1)
gpg.stop()

print("Stop the motors for 1 second.")
time.sleep(1)

print("Drive the motors 50 cm and then stop.")
gpg.drive_cm(50, True)
time.sleep(1)

print("Turn right 1 second.")
gpg.right()
time.sleep(1)

print("Turn left 1 second.")
gpg.left()
time.sleep(1)

print("Stop!")
gpg.stop()
print("Done!")
gpg = EasyGoPiGo3()
gpg.forward()
time.sleep(1)
gpg.stop()

#while True:
 #   easygopigo3.EasyGoPiGo3.drive_cm(5.0,blocking = True)
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

""""
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfir((x1,x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
"""


def canny(image):
    #this turns the image grey
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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

def regionOfInterest(image):
    amountTaken = 60
    rectangle = np.array([
        [(0,imageHeight), (0, amountTaken), (imageWidth,amountTaken), (imageWidth, imageHeight)]
        ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, rectangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


#this is an array made from the image
#image = cv2.imread('image.jpg')
#this is a copy of the array above
lane_image = np.copy(image)

canny_image = canny(lane_image)
croppedImage = regionOfInterest(canny_image)
lines = cv2.HoughLinesP(croppedImage, 2, np.pi/180,100, np.array([]), minLineLength=40,maxLineGap=5)
line_image = display_lines(lane_image,lines)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1,1)
cv2.imshow("asscum", combo_image)
cv2.waitKey(0)