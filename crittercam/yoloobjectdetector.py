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

import numpy as np
import cv2


class YOLOObjectDetector:
    def __init__(self, config):
        self.object_detected = False

        self.object_list = config.OBJECT_LIST
        self.scale_factor = config.YOLO_SCALEFACTOR
        self.kernel = config.YOLO_KERNEL
        self.confidence = config.YOLO_CONFIDENCE
        self.threshold = config.YOLO_THRESHOLD

        self.thickness = config.BB_THICKNESS
        self.font = config.BB_FONTFACE
        self.font_scale = config.BB_FONTSCALE
        self.text_color = config.BB_TEXTCOLOR
        self.text_thickness = config.BB_TEXTTHICKNESS

        self.labels = open(config.COCO_LABELS_PATH).read().strip().split('\n')
        self.colors = np.random.randint(
            100,
            255,
            size=(len(self.labels), 3),
            dtype='uint8'
        ).tolist()
        self.network = cv2.dnn.readNetFromDarknet(
            config.YOLO_CONFIG_PATH,
            config.YOLO_WEIGHTS_PATH
        )
        self.layer_names = None
        self.boxes = None
        self.confidences = None
        self.class_ids = None

    def get_layernames(self):
        self.layer_names = self.network.getLayerNames()
        self.layer_names = [
            self.layer_names[i[0] - 1] for i in
            self.network.getUnconnectedOutLayers()
        ]

    def preprocess_image(self, image):
        blob = cv2.dnn.blobFromImage(
            image,
            self.scale_factor,
            self.kernel,
            swapRB=True,
            crop=False
        )
        return blob

    def predict(self, blob):
        self.network.setInput(blob)
        if self.layer_names is None:
            self.get_layernames()
        return self.network.forward(self.layer_names)

    def prune(self, boxes, confidences):
        return cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.confidence,
            self.threshold
        )

    def detect_objects(self, image, height, width):
        blob = self.preprocess_image(image)
        outputs = self.predict(blob)

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)

                if self.labels[class_id] not in self.object_list:
                    continue

                confidence = scores[class_id]

                if confidence > self.confidence:
                    box = detection[0:4] * np.array(
                        [width, height, width, height]
                    )
                    box = box.astype('int')
                    (centerX, centerY, box_width, box_height) = box

                    x = int(centerX - (box_width / 2))
                    y = int(centerY - (box_height / 2))

                    boxes.append([x, y, int(box_width), int(box_height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

            indices = self.prune(boxes, confidences)
            if len(indices) > 0:
                self.object_detected = True
                indices = indices.flatten()
                boxes = [boxes[i] for i in indices]
                confidences = [confidences[i] for i in indices]
                class_ids = [class_ids[i] for i in indices]
            else:
                self.object_detected = False

        self.boxes = boxes
        self.confidences = confidences
        self.class_ids = class_ids

        return (boxes, confidences, class_ids)


def draw_boundingboxes(frame, object_detector):
    (boxes, confidences, class_ids) = object_detector.detect_objects(
        frame,
        frame.shape[0],
        frame.shape[1]
    )

    for i in range(len(boxes)):
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        color = object_detector.colors[class_ids[i]]

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            object_detector.thickness
        )

        label = f'{object_detector.labels[class_ids[i]]}: {confidences[i]:.4f}'
        ((lx, ly), _) = cv2.getTextSize(
            label,
            object_detector.font,
            object_detector.font_scale,
            object_detector.text_thickness
        )

        cv2.rectangle(
            frame,
            (x, y),
            (x + lx, y - ly),
            color,
            cv2.FILLED
        )

        cv2.putText(
            frame,
            label,
            (x, y),
            object_detector.font,
            object_detector.font_scale,
            object_detector.text_color,
            object_detector.text_thickness
        )
