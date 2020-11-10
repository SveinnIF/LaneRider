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
# Ideal dimensions of final image
TARGET_H = 200
TARGET_W = imageWidth


# Birdseye transform lookup table
src = np.float32([[0, TARGET_H], [TARGET_W, TARGET_H], [0, 0], [TARGET_W, 0]])
dst = np.float32([[165, TARGET_H], [475, TARGET_H], [0, 0], [TARGET_W, 0]])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

# IMAGE PROCESSING FUNCTIONS
# Functions that transform the input image
def birdsEyeTransform(image):

    img = image[280:(280+TARGET_H), 0:TARGET_W]
    warped_img = cv2.warpPerspective(img, M, (TARGET_W, TARGET_H))

    return warped_img


# def filterBlackWhite(image):
#
#     image_grey_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     (tresh, image_black_white) = cv2.threshold(image_grey_scale, 127, 255, cv2.THRESH_BINARY)
#     return image_black_white
#
def getCountorPts(image):
    #image_grey_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #ret, thresh = cv2.threshold(image, 127, 255, 0)
    im, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(contours)
    return contours
    # cnt = contours[0].reshape(-1, 2)
    #print(cnt)

    #
    # mask = np.zeros(image.shape, np.uint8)
    # cv2.drawContours(mask, contours, 0, 255, -1)
    # pixelpoints = np.transpose(np.nonzero(mask))
    # pixelpoints = cv2.findNonZero(mask)


def splitCoordinateArray(contours):

    x_arr = []
    y_arr = []

    for i in range(len(contours)):
        for j in range(len(contours[i])):
            x_arr.append(contours[i][j][0][0])
            y_arr.append(contours[i][j][0][1])
    return np.array(x_arr), np.array(y_arr)


def canny(image):
    #this turns the image grey
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #this blurs the image, making edge detection more reliable
    blur = cv2.GaussianBlur(gray, (5,5),0)
    #this derives the array and thereby detects the change in
    #intensity in nearby pixles
    canny = cv2.Canny(blur,155,255)
    return canny


# def convolutetheprogram(image):
#
#     KERNEL = [-1, -1, 2, 2,-1,-1]
#     out_image = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2GRAY)
#     for i in range(len(image)):
#        out_image[i] = np.convolve(KERNEL, out_image[i], "same")
#     return out_image


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
        #cropped_image = CropImageFromTop(lane_image, 60)
        #canny_image = canny(lane_image)
        img_canny = canny(lane_image)
        img_birdseye = birdsEyeTransform(img_canny)
        img_birdseye2 = birdsEyeTransform(lane_image)
        contours = getCountorPts(img_birdseye)
        mask = np.zeros(image.shape, np.uint8)
        cv2.drawContours(img_birdseye2, contours, 0, 255, -1)
        if len(contours) > 0:
            x_arr, y_arr = splitCoordinateArray(contours)
            pol1 = np.polyfit(x_arr, y_arr, 2)
            print(pol1)
        #img_convolutional = convolutetheprogram(img_birdseye2)
        # getCountorPts(img_birdseye)
        #img_blackwhite = filterBlackWhite(img_birdseye)
        #cntPts = getCountorPts(img_canny)
        cv2.imshow("lineVision", img_canny)
        cv2.imshow("lineVision1", img_birdseye)
        #cv2.imshow("lineVision2", img_convolutional)
        cv2.waitKey(1)


























