import io
import time
import threading
import picamera
import cv2
import numpy as np

class ImageProcessor(threading.Thread):
    def __init__(self, owner):
        super(ImageProcessor, self).__init__()
        self.stream = io.BytesIO()
        self.event = threading.Event()
        self.terminated = False
        self.owner = owner
        self.start()

    def run(self):
        # This method runs in a separate thread
        while not self.terminated:
            # Wait for an image to be written to the stream
            if self.event.wait(1):
                try:
                    self.stream.seek(0)
                    # Read the image and do some processing on it
                    data = np.fromstring(self.stream.getvalue(), dtype=np.uint8)
                    # "Decode" the image from the array, preserving colour
                    image = cv2.imdecode(data, 1)

                    def canny(image):
                        # this turns the image grey
                        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                        # this blurs the image, making edge detection more reliable
                        blur = cv2.GaussianBlur(gray, (5, 5), 0)
                        # this derives the array and thereby detects the change in
                        # intensity in nearby pixles
                        canny = cv2.Canny(blur, 50, 150)
                        return canny

                    def display_lines(image, lines):
                        line_image = np.zeros_like(image)
                        if lines is not None:
                            for line in lines:
                                x1, y1, x2, y2 = line.reshape(4)
                                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
                                print((x1, y1), (x2, y2))
                        return line_image

                    def regionOfInterest(image):
                        amountTaken = 60
                        rectangle = np.array([
                            [(0, imageHeight), (0, amountTaken), (imageWidth, amountTaken), (imageWidth, imageHeight)]
                        ])
                        mask = np.zeros_like(image)
                        cv2.fillPoly(mask, rectangle, 255)
                        masked_image = cv2.bitwise_and(image, mask)
                        return masked_image

                    # this is an array made from the image
                    # image = cv2.imread('image.jpg')
                    # this is a copy of the array above
                    lane_image = np.copy(image)

                    canny_image = canny(lane_image)
                    croppedImage = regionOfInterest(canny_image)
                    lines = cv2.HoughLinesP(croppedImage, 2, np.pi / 180, 100, np.array([]), minLineLength=40,
                                            maxLineGap=5)
                    line_image = display_lines(lane_image, lines)
                    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
                    cv2.imshow("asscum", combo_image)
                    cv2.waitKey(0)

                    #Image.open(self.stream)
                    #...
                    #...
                    # Set done to True if you want the script to terminate
                    # at some point
                    #self.owner.done=True
                finally:
                    # Reset the stream and event
                    self.stream.seek(0)
                    self.stream.truncate()
                    self.event.clear()
                    # Return ourselves to the available pool
                    with self.owner.lock:
                        self.owner.pool.append(self)

class ProcessOutput(object):
    def __init__(self):
        self.done = False
        # Construct a pool of 4 image processors along with a lock
        # to control access between threads
        self.lock = threading.Lock()
        self.pool = [ImageProcessor(self) for i in range(4)]
        self.processor = None

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame; set the current processor going and grab
            # a spare one
            if self.processor:
                self.processor.event.set()
            with self.lock:
                if self.pool:
                    self.processor = self.pool.pop()
                else:
                    # No processor's available, we'll have to skip
                    # this frame; you may want to print a warning
                    # here to see whether you hit this case
                    self.processor = None
        if self.processor:
            self.processor.stream.write(buf)

    def flush(self):
        # When told to flush (this indicates end of recording), shut
        # down in an orderly fashion. First, add the current processor
        # back to the pool
        if self.processor:
            with self.lock:
                self.pool.append(self.processor)
                self.processor = None
        # Now, empty the pool, joining each thread as we go
        while True:
            with self.lock:
                try:
                    proc = self.pool.pop()
                except IndexError:
                    pass # pool is empty
            proc.terminated = True
            proc.join()

with picamera.PiCamera(resolution='VGA') as camera:
    camera.start_preview()
    time.sleep(2)
    output = ProcessOutput()
    camera.start_recording(output, format='mjpeg')
    while not output.done:
        camera.wait_recording(1)
    camera.stop_recording()