import cv2 as cv
import numpy as np
import time
import select
import sys
from time import sleep
from picamera import PiCamera
import picamera.array
from easygopigo3 import EasyGoPiGo3

GPG = EasyGoPiGo3()
cam = PiCamera()

# -----CONFIG-----

# Camera
WIDTH = 128
HEIGHT = 80
# Canny
CANNY_THRESHOLD_1 = 50
CANNY_THRESHOLD_2 = 150
# HoughLines
RHO_RESOLUTION = 1
THETA_RESOLUTION = np.pi / 180
VOTE_THRESHOLD = 5
MIN_LINE_LENGTH = 15
MAX_LINE_GAP = 20
# Motion control
FORWARD_POWER = 50


# -----FUNCTIONS-----

def draw_hough_lines(hough_lines, img):
    for line in hough_lines:
        if line == None: break
        color = (0, 0, 0)
        x1, y1, x2, y2, _, _, edge_type = line
        if edge_type == 'intersection_top':
            color = (255, 0, 0)
        elif edge_type == "intersection_bottom":
            color = (0, 255, 0)
        elif edge_type == "right":
            color = (100, 0, 150)
        else:
            color = (0, 150, 255)
        cv.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)


def draw_control_line(control_line, img):
    x1, y1, x2, y2, _, _ = control_line
    cv.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)


def get_control_values(line):
    x1, y1, x2, y2, slope, b = line
    opposite_edge = (y1 - y2)
    adjacent_edge = (x2 - x1)
    hypotenus = np.sqrt(np.square(opposite_edge) + np.square(adjacent_edge))
    sin = opposite_edge / hypotenus
    cos = adjacent_edge / hypotenus
    theta = np.angle(complex(cos, sin), deg=True)
    return (cos, sin, theta)


# Stops program when enter-key is pressed
def stop_program():
    i, o, e = select.select([sys.stdin], [], [], 0.0001)
    for s in i:
        if s == sys.stdin:
            input = sys.stdin.readline(1)
            return True
    return False


# Initializes camera
def cam_init(w, h):
    cam.resolution = (w, h)
    cam.framerate = 60
    cam.start_preview()
    time.sleep(1)


# Captures a bgr image for use in image processing
def get_bgr_image():
    with picamera.array.PiRGBArray(cam) as stream:
        cam.capture(stream, format="bgr", use_video_port=True)
        IMG = stream.array
        return IMG


# Returns the hough lines in a bgr image
def get_hough_lines(img):
    global gray_scale

    gray_scale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    edges = cv.Canny(gray_scale, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2, apertureSize=3)

    roi = region_of_interest(edges)

    hough_lines = cv.HoughLinesP(roi, rho=RHO_RESOLUTION, theta=THETA_RESOLUTION, threshold=VOTE_THRESHOLD,
                                 maxLineGap=MAX_LINE_GAP, minLineLength=MIN_LINE_LENGTH)
    return hough_lines


# Removes unnessecary array layer and add slope and y-axis intersection line data
def format_lines(lines):
    formatted_array = []
    if len(lines) == 0:
        return formatted_array
    else:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if y2 > y1:
                x2, y2, x1, y1 = line[0]
            if x1 == x2:
                x2 += 1
            slope = round(float((y2 - y1) / (x2 - x1)), 3)
            y_intersect = int(y1 - slope * x1)
            edge_type = get_edge_type(x1, y1, x2, y2, slope, gray_scale)
            if edge_type == "throw":
                continue
            formatted_array.append([x1, y1, x2, y2, slope, y_intersect, edge_type])
        return formatted_array


# Returns a string telling if the edge is on the left or right of the lane, or if it is an intersection edge
def get_edge_type(x1, y1, x2, y2, slope, gray_scale):
    x0 = int(np.mean([x1, x2]))
    y0 = int(np.mean([y2, y1]))
    if (slope < 0.1 and slope >= 0) or (slope > -0.1 and slope <= 0):
        if y0 + 5 > len(gray_scale) - 1:
            if gray_scale[y0 - 5][x0] > 100:
                return "intersection_bottom"
            return "intersection_top"
        if y0 - 5 < 0:
            if gray_scale[y0 + 5][x0] > 100:
                return "intersection_top"
            return "intersecton_bottom"
        over_edge = gray_scale[y0 - 5][x0]
        below_edge = gray_scale[y0 + 5][x0]
        if over_edge < below_edge:
            return "intersection_top"
        return "intersection_bottom"
    if x0 - 5 < 0:
        if gray_scale[y0][x0 + 5] > 100:
            return "left"
        return "right"
    if x0 + 5 > len(gray_scale[0]) - 1:
        if gray_scale[y0][x0 - 5] > 100:
            return "right"
        return "left"
    left_of_edge = gray_scale[y0][x0 - 5]
    right_of_edge = gray_scale[y0][x0 + 5]
    if left_of_edge < right_of_edge:
        return "left"
    return "right"


# Filters out any hough line not part of the bottom 2
def get_bottom_lines(lines):
    bottom_lines = []
    y_values = []
    for line in lines:
        y_values.append(line[1])
        y_values.append(line[3])
    y_values.sort(reverse=True)
    y_values = [y_values[0], y_values[1]]
    for line in lines:
        if line[1] in y_values or line[3] in y_values:
            if len(bottom_lines) != 0:
                for bottom_line in bottom_lines:
                    if not lines_equivalent(line, bottom_line):
                        bottom_lines.append(line)
            else:
                bottom_lines.append(line)
    return bottom_lines


# Returns a line pointing in the direction the robot should turn
def get_control_line(lines):
    heading_x1 = WIDTH / 2
    heading_y1 = int(HEIGHT)

    if len(lines) == 1:
        line = lines[0]
        slope = line[4]
        heading_b = int(heading_y1 - slope * heading_x1)
        heading_y2 = min([line[1], line[3]])
        if slope < 0:
            heading_x2 = WIDTH
        else:
            heading_x2 = 0
        global last_found_edge
        last_found_edge = line[6]
        return [heading_x1, heading_y1, heading_x2, heading_y2, slope, heading_b]

    y2_array = []
    x2_array = []
    for line in lines:
        if line == []:
            continue
        x1, y1, x2, y2, _, _, _ = line
        if y2 < y1:
            y2_array.append(y2)
            x2_array.append(x2)
        else:
            y2_array.append(y1)
            x2_array.append(x1)
    heading_y2 = np.min(y2_array)
    heading_x2 = np.mean(x2_array)

    return [int(heading_x1), int(heading_y1), int(heading_x2), int(heading_y2), lines[0][4],
            int(heading_y1 - lines[0][4] * heading_x1)]


# code taken from towardsdatascience.com/deeppicar-part4-lane-following-via-opencv-737dd9e47c96
def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (0, height * 2 / 3),
        (width, height * 2 / 3),
        (width, height),
        (0, height),
    ]], np.int32)

    cv.fillPoly(mask, polygon, 255)
    cropped_edges = cv.bitwise_and(edges, mask)
    return cropped_edges


def get_filtered_lines(lines):
    right_edge_lines = []
    left_edge_lines = []
    best_edges = []

    best_right_edge = None
    best_left_edge = None

    for line in lines:
        if line[6] == "right":
            right_edge_lines.append(line)
        elif line[6] == "left":
            left_edge_lines.append(line)

    best_edge_info = [None, None]

    # Find best right edge
    if len(right_edge_lines) != 0:
        for i, line in enumerate(right_edge_lines):
            distance_to_line = distance_from_corner_to_edge(line)
            if i == 0:
                best_edge_info = [i, distance_to_line]
            elif distance_to_line < best_edge_info[1]:
                best_edge_info = [i, distance_to_line]
        best_right_edge = right_edge_lines[best_edge_info[0]]

    # Find best left edge
    if len(left_edge_lines) != 0:
        for i, line in enumerate(left_edge_lines):
            distance_to_line = distance_from_corner_to_edge(line)
            if i == 0:
                best_edge_info = [i, distance_to_line]
            elif distance_to_line < best_edge_info[1]:
                best_edge_info = [i, distance_to_line]
        best_left_edge = left_edge_lines[best_edge_info[0]]

    if best_right_edge is not None:
        best_edges.append(best_right_edge)
    if best_left_edge is not None:
        best_edges.append(best_left_edge)
    print(best_edges)
    return best_edges


# Find shortest distance from bottom right corner to edge line
def distance_from_corner_to_edge(line):
    corner_x = WIDTH
    corner_y = HEIGHT
    x0 = line[0]
    y0 = line[1]
    adjacent = corner_x - x0
    opposite = corner_y - y0
    distance_to_line = np.sqrt(np.square(adjacent) + np.square(opposite))
    return distance_to_line


def wheel_control(turn_rate):
    speed_percentage = 25

    left_power = FORWARD_POWER + turn_rate * speed_percentage
    right_power = FORWARD_POWER - turn_rate * speed_percentage
    # print("Left Power: {}, Right Power: {}".format(left_power,right_power))

    GPG.set_motor_power(GPG.MOTOR_LEFT, left_power)
    GPG.set_motor_power(GPG.MOTOR_RIGHT, right_power)


def detect_intersection(lines, filtered_lines):
    # global intersection_counter
    slope_treshold = 5
    intersection_line_exist = False
    is_intersection = False
    for line in lines:
        if line[6] == "intersection_bottom":
            print("Intersection lines detected")
            intersection_line_exist = True
            break

    if intersection_line_exist:
        for line in filtered_lines:
            if abs(line[5]) > slope_treshold:
                print("Intersection detected")
                is_intersection = True
                break
    # Check if left or right intersection
    if is_intersection:
        for line in lines:
            if line[6] == "intersection_bottom":
                for filtered_line in filtered_lines:
                    if len(filtered_line) != 0:
                        if np.max([line[2], line[0]]) > filtered_line[2]:
                            if (line[2] - line[0] > WIDTH / 5):
                                if filtered_line[6] == "right":
                                    print("Right intersection")

                                    return True

    return False


# -----INIT---------

print("Initializing...")

cam_init(WIDTH, HEIGHT)

print("Ready!")
GPG.set_speed(250)
GPG.forward()
last_found_edge = None

# -----LOOP---------


while True:
    if stop_program():
        GPG.set_motor_power(GPG.MOTOR_LEFT + GPG.MOTOR_RIGHT, 0)
        break
    print("----------------\n")

    img = get_bgr_image()

    lines = get_hough_lines(img)

    if lines is None:
        if last_found_edge != None:
            # print("DWADWADWDWAW")
            if last_found_edge == "right":
                wheel_control(-0.9)
            elif last_found_edge == "left":
                wheel_control(0.9)
        continue

    formatted_lines = format_lines(lines)

    filtered_lines = get_filtered_lines(formatted_lines)

    if detect_intersection(formatted_lines, filtered_lines):
        GPG.drive_cm(15, True)
        GPG.turn_degrees(90)
        continue

    control_line = get_control_line(filtered_lines)

    (cos, sin, theta) = get_control_values(control_line)

    wheel_control(cos)

cam.stop_preview()