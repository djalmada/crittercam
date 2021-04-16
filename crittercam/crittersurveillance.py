from crittercam import (
    EventRecorder,
    MotionDetector,
    TwilioNotifier,
    YOLOObjectDetector,
    draw_boundingboxes
)
import cv2
import datetime
import imagezmq
import importlib
import os


def main():
    config = importlib.import_module('crittercam.config')

    image_hub = imagezmq.ImageHub()
    motion_detector = MotionDetector(config)
    total_frames = 0

    object_detector = YOLOObjectDetector(config)
    event_recorder = EventRecorder(config.ER_BUFFERSIZE, config.ER_TIMEOUT)
    notifier = TwilioNotifier(config)

    last_notif_time = datetime.datetime(1970, 1, 1)

    while True:
        (host_name, frame) = image_hub.recv_image()
        image_hub.send_reply(b'Connection established')

        if frame is None:
            continue

        # Process frames from client for motion detection
        (height, width) = frame.shape[:2]
        ratio = config.FRAME_WIDTH / float(width)
        dsize = (config.FRAME_WIDTH, int(height * ratio))
        frame = cv2.resize(frame, dsize=dsize, interpolation=cv2.INTER_AREA)

        motion_detected = motion_detector.detect_motion(frame)

        # If motion detected perform YOLO object detection
        if motion_detected:
            timestamp = datetime.datetime.now()

            if config.ANNOTATE_TIMESTAMP:
                cv2.putText(
                    frame,
                    timestamp.strftime('%A %d %B %Y %I:%M:%S%p'),
                    (10, frame.shape[0] - 10),
                    config.TS_FONTFACE,
                    config.TS_FONTSCALE,
                    config.TS_TEXTCOLOR,
                    config.TS_TEXTTHICKNESS
                )

            if config.DRAW_BOUNDING_BOXES:
                draw_boundingboxes(frame, object_detector)

            if object_detector.object_detected:

                if config.SEND_NOTIFICATIONS:
                    current_time = datetime.datetime.now()
                    since_last_notif = \
                        (current_time - last_notif_time).total_seconds()

                    if since_last_notif >= config.NOTIFICATION_DELAY:
                        objects = [object_detector.labels[i] for i in
                                   object_detector.class_ids]
                        notification = '{} detected on {} camera'.format(
                            objects,
                            host_name
                        )
                        notifier.send_sms(notification)
                        last_notif_time = datetime.datetime.now()

                if not event_recorder.is_recording:
                    date = timestamp.strftime('%Y-%m-%d')
                    os.makedirs(
                        os.path.join(
                            config.RECORDING_OUTPUT_PATH,
                            date
                        ),
                        exist_ok=True
                    )
                    recording_path = '{}/{}/{}.avi'.format(
                        config.RECORDING_OUTPUT_PATH,
                        date,
                        timestamp.strftime('%H%M%S')
                    )
                    event_recorder.start(
                        recording_path,
                        config.VIDEO_CODEC,
                        config.RECORDING_FPS
                    )

                if config.SAVE_SCREENSHOTS:
                    date = timestamp.strftime('%Y-%m-%d')
                    os.makedirs(
                        os.path.join(
                            config.SCREENSHOT_OUTPUT_PATH,
                            date
                        ),
                        exist_ok=True
                    )
                    screenshot_path = '{}/{}/{}.png'.format(
                        config.SCREENSHOT_OUTPUT_PATH,
                        date,
                        timestamp.strftime('%H%M%S')
                    )
                    event_recorder.take_screenshot(
                        frame,
                        screenshot_path,
                        config.SCREENSHOT_DELAY
                    )

            event_recorder.update(frame)

            if event_recorder.is_recording and \
                    event_recorder.consecutive_frames == config.ER_BUFFERSIZE:
                event_recorder.stop()

        total_frames += 1

        if config.DISPLAY:
            cv2.imshow('Video Stream', frame)
            key = cv2.waitKey(1) & 0xff

            if key == ord('q'):
                break

    # Perform cleanup
    if event_recorder.is_recording:
        event_recorder.stop()

    if config.DISPLAY:
        cv2.destroyAllWindows()
