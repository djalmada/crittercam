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

from collections import deque
from queue import Queue
from threading import Thread
import numpy as np
import cv2
import time


class EventRecorder:
    def __init__(self, buffer_size=64, timeout=1.0):
        self.buffer_size = buffer_size
        self.timeout = timeout

        self.frame_buffer = deque(maxlen=buffer_size)
        self.frame_queue = None
        self.is_recording = None
        self.thread = None
        self.writer = None

        self.frames_without_motion = 0
        self.consecutive_frames = 0
        self.frames_since_screenshot = np.inf

    def start(self, output_path, video_codec, fps):
        self.is_recording = True
        self.frame_queue = Queue()
        (height, width, _) = self.frame_buffer[0].shape
        self.writer = cv2.VideoWriter(
            output_path,
            video_codec,
            fps,
            (height, width)
        )

        for i in range(len(self.frame_buffer), 0, -1):
            self.frame_queue.put(self.frame_buffer[i - 1])

        self.thread = Thread(target=self.record_video, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self, frame):
        '''
        '''
        self.frame_buffer.appendleft(frame)

        if self.is_recording:
            self.frame_queue.put(frame)
            self.consecutive_frames += 1

    def record_video(self):
        while True:
            if not self.is_recording:
                return

            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                self.writer.write(frame)

            else:
                time.sleep(self.timeout)

    def take_screenshot(self, image, screenshot_path, delay=30):
        if self.frames_since_screenshot >= delay:
            cv2.imwrite(screenshot_path, image)
            self.frames_since_screenshot = 0

        self.frames_since_screenshot += 1

    def stop(self):
        self.is_recording = False
        self.consecutive_frames = 0
        self.thread.join()
        while not self.frame_queue.empty():
            frame = self.frame_queue.get()
            self.writer.write(frame)
        self.writer.release()
