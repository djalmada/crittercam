import cv2

# Video capture settings
PICAMERA = False
FRAME_WIDTH = 480
CAMERA_FPS = 30
DISPLAY = True

# Motion detection settings
BACKGROUNDSUB_FRAMES = 256
DISTANCE_TO_THRESHOLD = 16
DETECT_SHADOWS = False

# Object detection settings
COCO_LABELS_PATH = 'yolo-coco/coco.names'
YOLO_CONFIG_PATH = 'yolo-coco/yolov4/yolov4.cfg'
YOLO_WEIGHTS_PATH = 'yolo-coco/yolov4/yolov4.weights'
YOLO_SCALEFACTOR = 1.0 / 255.0
YOLO_KERNEL = (704, 704)
YOLO_CONFIDENCE = 0.5
YOLO_THRESHOLD = 0.3
YOLO_FPS = 10
OBJECT_LIST = [
    # 'bear',
    'bird',
    'cat',
    # 'cow',
    'dog',
    # 'elephant',
    # 'giraffe',
    # 'horse',
    'person',
    # 'sheep',
    # 'zebra',
]

# Bounding box settings
DRAW_BOUNDING_BOXES = True
BB_THICKNESS = 4
BB_FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
BB_FONTSCALE = 0.75
BB_TEXTCOLOR = (0, 0, 0)
BB_TEXTTHICKNESS = 2

# Timestamp settings
ANNOTATE_TIMESTAMP = True
TS_FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
TS_FONTSCALE = 0.35
TS_TEXTCOLOR = (0, 0, 255)
TS_TEXTTHICKNESS = 1

# Event recorder settings
ER_BUFFERSIZE = 64
ER_TIMEOUT = 1.0

# Screenshot settings
SAVE_SCREENSHOTS = True
SCREENSHOT_OUTPUT_PATH = 'output/images'
SCREENSHOT_DELAY = 30  # in frames

# Video recording settings
RECORDING_OUTPUT_PATH = 'output/videos'
VIDEO_CODEC = cv2.VideoWriter_fourcc(*'MJPG')
RECORDING_FPS = 30

# Twilio settings
SEND_NOTIFICATIONS = False
NOTIFICATION_DELAY = 30  # in seconds
TWILIO_SID = 'YOUR_TWILIO_SID'
TWILIO_AUTH = 'YOUR_TWILIO_AUTH_ID'
TWILIO_TO = 'YOUR_PHONE_NUMBER'
TWILIO_FROM = 'YOUR_TWILIO_PHONE_NUMBER'
