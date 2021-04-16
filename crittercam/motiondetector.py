'''The MIT License (MIT)

Copyright (c) 2021, Demetrius Almada

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.'''

import cv2


class MotionDetector:
    def __init__(self, config):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            config.BACKGROUNDSUB_FRAMES,
            config.DISTANCE_TO_THRESHOLD,
            config.DETECT_SHADOWS
        )
        self.foreground_mask = None

    def remove_noise(self,):
        self.forground_mask = cv2.threshold(
            self.foreground_mask,
            0,
            255,
            cv2.THRESH_BINARY
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (9, 9)
        )

        self.foreground_mask = cv2.erode(
            self.foreground_mask,
            kernel,
            1
        )

        self.foreground_mask = cv2.dilate(
            self.foreground_mask,
            kernel,
            2
        )

    def detect_motion(self, frame):
        self.foreground_mask = self.subtractor.apply(frame)

        if self.foreground_mask is None:
            return False

        self.remove_noise()

        contours = cv2.findContours(
            self.foreground_mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        motion_detected = len(contours[0]) > 0
        if motion_detected:
            return True
        else:
            return False
