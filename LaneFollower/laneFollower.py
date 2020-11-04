import cv2
import numpy as np
import picamera
import matplotlib.pyplot as plt
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

# code for image transform taken from: (And reconfigured)
# https://nikolasent.github.io/opencv/2017/05/07/Bird's-Eye-View-Transformation.html

def birdsEyeTransform(image, cropTop, cropBottom):

    src = np.float32([[0, imageHeight], [480, imageHeight], [0, 0], [imageWidth, 0]])
    dst = np.float32([[200, imageHeight], [280, imageHeight], [0, 0], [imageWidth, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    img = image[cropBottom:imageHeight, 0:imageWidth]
    warped_img = cv2.warpPerspective(img, M, (imageWidth, imageHeight))
    return warped_img

#Camera Values
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
        #cropped_image = CropImageFromTop(lane_image, 60)
        #print(cropped_image.shape)
        #canny_image = canny(lane_image)
        #print(waypoints)
        #print(lines)
        img_birdseye = birdsEyeTransform(lane_image, 320, 0)
        cv2.imshow("lineVision", img_birdseye)
        cv2.waitKey(1)


























