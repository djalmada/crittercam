#!/usr/bin/env python
from imutils.video import VideoStream
import argparse
import imagezmq
import socket
import time

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--server-ip', required=True,
                    help='IP address of server to which client will connect')
parser.add_argument('-p', '--pi-camera', type=bool, default=True,
                    help='Toggle use of Raspberry Pi camera module')
args = vars(parser.parse_args())

sender = imagezmq.ImageSender(connect_to=f'tcp://{args["server_ip"]}:5555')

host_name = socket.gethostname()
if args['pi_camera']:
    vs = VideoStream(usePiCamera=True).start()
else:
    vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    sender.send_image(host_name, frame)
