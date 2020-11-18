import cv2
import numpy as np
import picamera
import math
from easygopigo3 import EasyGoPiGo3
GPG = EasyGoPiGo3()


imageHeight = 480
imageWidth = 640

TARGET_H = 200
TARGET_W = imageWidth

# Birdseye transform lookup table
src = np.float32([[0, TARGET_H], [TARGET_W, TARGET_H], [0, 0], [TARGET_W, 0]])
dst = np.float32([[165, TARGET_H], [475, TARGET_H], [0, 0], [TARGET_W, 0]])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

def birdsEyeTransform(image):

    img = image[280:(280+TARGET_H), 0:TARGET_W]
    warped_img = cv2.warpPerspective(img, M, (TARGET_W, TARGET_H))

    return warped_img

def stop_program():
    i, o, e = select.select([sys.stdin], [], [], 0.0001)
    for s in i:
        if s == sys.stdin:
            input = sys.stdin.readline(1)
            return True
    return False

def canny(image):
    #this turns the image grey
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

def find_usable_line(lines):
    previous_point = []
    lowest_value = []
    if lines is None:
        print('piss')
        return 90 # Change at a later time
    for line in lines:
        if lines is None:
            return 90
        x1, y1, x2, y2 = line.reshape(4)
        if previous_point != []:
            if previous_point[0] > x1 and previous_point[1] > y1:
                lowest_value = [x1, y1, x2, y2]
                print('lowest value in if ')
                print(lowest_value)
        previous_point = [x1, y1]
    print('lowest value final ')
    print(lowest_value)
    if lowest_value == []:
        return 90
    katet = lowest_value[2] - lowest_value[0]

    motKat = lowest_value[3] - lowest_value[1]
    hypotenus = math.sqrt(math.pow(math.fabs(katet),2) + math.pow(math.fabs(motKat),2))
    angleRad = math.acos(katet/hypotenus)
    angle = angleRad * 180 / math.pi
    return angle


def wheel_control(turn_rate):
    speed_percentage = 25
    FORWARD_POWER = 50

    left_power = FORWARD_POWER + turn_rate * speed_percentage
    right_power = FORWARD_POWER - turn_rate * speed_percentage
    # print("Left Power: {}, Right Power: {}".format(left_power,right_power))

    GPG.set_motor_power(GPG.MOTOR_LEFT, left_power)
    GPG.set_motor_power(GPG.MOTOR_RIGHT, right_power)

def waypoint_detection(image):
    NUM_IGNORED_ROWS = 120
    NUM_WAYPOINTS = 3

    num_rows = len(image)
    num_relevant_rows = (num_rows - NUM_IGNORED_ROWS)


    step = num_relevant_rows // (NUM_WAYPOINTS + 1)
    start = NUM_IGNORED_ROWS + step
    waypoints = []
    for i in range(start, num_rows, step):
        print(i)
        for j in range(len(image[i]) - 1, -1, -1):
            if image[i][j] == 255:
                waypoints.append((i, j))
                break
    return waypoints


def regionOfInterest(image):
    amountTaken = 150
    rectangle = np.array([
        [(0,imageHeight), (0, amountTaken), (imageWidth,amountTaken), (imageWidth, imageHeight)]
        ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, rectangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


with picamera.PiCamera() as camera:
    camera.resolution = (imageWidth,imageHeight)
    camera.framerate = 30
    image = np.empty((imageHeight * imageWidth * 3), dtype=np.uint8)
    image = image.reshape((imageHeight,imageWidth,3))
    for frame in camera.capture_continuous(image, format="bgr", use_video_port=True):
        if stop_program():
            GPG.set_motor_power(GPG.MOTOR_LEFT + GPG.MOTOR_RIGHT, 0)
            break
        print("----------------\n")
        lane_image = np.copy(image)
        canny_image = canny(lane_image)
        #img_birdseye = birdsEyeTransform(canny_image)
        croppedImage = regionOfInterest(canny_image)
        #waypoints = waypoint_detection(croppedImage)
        #print(waypoints)
        lines = cv2.HoughLinesP(croppedImage, 2, np.pi / 180, 100, np.array([]), minLineLength=20, maxLineGap=15)
        #print(lines)
        wheel_control(find_usable_line(lines))
        line_image = display_lines(lane_image, lines)
        combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
        print(find_usable_line(lines))
        cv2.imshow("lineVision",combo_image)
        cv2.waitKey(1)
