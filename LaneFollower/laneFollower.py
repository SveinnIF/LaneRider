import cv2
import numpy as np
import picamera
import matplotlib.pyplot as plt
from easygopigo3 import EasyGoPiGo3
GPG = EasyGoPiGo3()


# code for image transform taken from: (And reconfigured)
# https://nikolasent.github.io/opencv/2017/05/07/Bird's-Eye-View-Transformation.html

# VARIABLES
# Camera Values
imageHeight = 480
imageWidth = 640
# image cropping values
cropTop = 200
cropBottom = imageHeight - 0
croppedHeight = cropBottom - cropTop

# Birdseye transform lookup table
src = np.float32([[0, croppedHeight], [640, croppedHeight], [0, 0], [imageWidth, 0]])
dst = np.float32([[imageWidth / 2 - 35, croppedHeight], [imageWidth / 2 + 35, croppedHeight], [0, 0], [imageWidth, 0]])
M = cv2.getPerspectiveTransform(src, dst)

# IMAGE PROCESSING FUNCTIONS
# Functions that transform the input image
def birdsEyeTransform(image):

    img = image[cropTop:croppedHeight, 0:imageWidth]
    warped_img = cv2.warpPerspective(img, M, (imageWidth, croppedHeight))

    return warped_img


# def filterBlackWhite(image):
#
#     image_grey_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     (tresh, image_black_white) = cv2.threshold(image_grey_scale, 127, 255, cv2.THRESH_BINARY)
#     return image_black_white



def getCountorPts(image):
    ret, thresh = cv2.threshold(image_gray, 127, 255, 0)
    im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0].reshape(-1, 2)



    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [cnt], 0, 255, -1)
    #pixelpoints = np.transpose(np.nonzero(mask))
    #pixelpoints = cv2.findNonZero(mask)


def canny(image):
    #this turns the image grey
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #this blurs the image, making edge detection more reliable
    blur = cv2.GaussianBlur(gray, (5,5),0)
    #this derives the array and thereby detects the change in
    #intensity in nearby pixles
    canny = cv2.Canny(blur,50,150)
    return canny

# MOTION CONTROL FUNCTIONS
# Functions that move or determine the path of the robot

# def unicycleSteering(velocity, turn_angle):
#
#
#     set_motor_power(MOTOR_RIGHT, )
#     set_motor_power(MOTOR_LEFT, )



with picamera.PiCamera() as camera:
    camera.resolution = (imageWidth,imageHeight)
    camera.framerate = 30
    image = np.empty((imageHeight * imageWidth * 3), dtype=np.uint8)
    image = image.reshape((imageHeight,imageWidth,3))
    for frame in camera.capture_continuous(image, format="bgr", use_video_port=True):
        lane_image = np.copy(image)
        #print(lane_image.shape)
        #cropped_image = CropImageFromTop(lane_image, 60)
        #print(cropped_image.shape)
        #canny_image = canny(lane_image)
        #print(waypoints)
        #print(lines)
        img_birdseye = birdsEyeTransform(lane_image)
        img_blackwhite = filterBlackWhite(img_birdseye)
        img_canny = canny(img_blackwhite)
        cntPts = getCountorPts(img_canny)
        #findPts(img_canny)
        #print(img_birdseye.shape)
        cv2.imshow("lineVision", img_blackwhite)
        cv2.waitKey(1)


























