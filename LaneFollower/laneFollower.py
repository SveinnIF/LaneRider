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

def birdsEyeTransform(image):
    #Crop image

    img = image[cropTop:croppedHeight, 0:imageWidth]
    warped_img = cv2.warpPerspective(img, M, (imageWidth, croppedHeight))

    return warped_img

def filterBlackWhite(image):
    image_grey_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (tresh, image_black_white) = cv2.threshold(image_grey_scale, 127, 255, cv2.THRESH_BINARY)
    return image_black_white

def findPts(image):
    pts = []
    step = -1
    for y in range(len(image), 0, step):
        for x in range(len(image[y]), 0, step):
            if image[y][x][0] > image[y][x-1][0]:
                pts.append((y, x))
                cv2.rectangle(image, (y, x), (y+1, x+1), 1)


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
        findPts(img_blackwhite)
        print(img_birdseye.shape)
        cv2.imshow("lineVision", img_blackwhite)
        cv2.waitKey(1)


























