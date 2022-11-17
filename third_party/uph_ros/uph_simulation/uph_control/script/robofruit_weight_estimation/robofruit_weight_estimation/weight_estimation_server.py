## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import time

import pyrealsense2.pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
#import cv2
from pyntcloud import PyntCloud
import os
import sys
import socket
import pickle
import struct


# Create a pipeline
from predict_weight import predict_weight

pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)


# Socket Create
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_name = socket.gethostname()
host_ip = socket.gethostbyname(host_name)
print('HOST IP:', host_ip)
port = 9999
# host_ip = '192.178.4.5'
socket_address = (host_ip, port)

# Socket Bind
server_socket.bind(socket_address)

# Socket Listen
server_socket.listen(5)
print("LISTENING AT:", socket_address)

# som vars
data = b""
payload_size = struct.calcsize("Q")

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)
# exit()

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

image_id = 1
# group_id = 0
canvas_directory = ''

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1.0  # 20 cms
clipping_distance = clipping_distance_in_meters / depth_scale

group_id = 'prediction'

# latency = 10
# counter = 1
# Streaming loop
while True:

    try:

        # wait for client connection
        print('wait for connection: ')
        client_socket, addr = server_socket.accept()

        while True:

            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Remove background - Set pixels further than clipping_distance to grey
            grey_color = 153
            depth_image_3d = np.dstack((depth_image, depth_image, depth_image)) #depth image is 1 channel, color is 3 channels
            bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

            # coloriser
            colourised_depth = np.asarray(rs.colorizer().colorize(aligned_depth_frame).get_data())
            # print(colourised_depth.shape)

            # Render images
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # images = np.hstack((bg_removed, colourised_depth, color_image))
            images = np.hstack((bg_removed, colourised_depth, color_image))
            # cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Align Example', images)
            key = cv2.waitKey(1)

            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                raise KeyboardInterrupt

            # s for save
            # elif key == 115:

            to_panda = predict_weight(color_image, depth_image)
            # to_panda = predict_weight(color_image, depth_image)

            to_panda = pickle.dumps(to_panda, protocol=2)
            to_panda = struct.pack("Q", len(to_panda)) + to_panda
            client_socket.send(to_panda)

            if to_panda != 0:
                time.sleep(10)

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        pipeline.stop()
        client_socket.close()
        server_socket.close()
        sys.exit()

    except struct.error as e:
        print('Exception: ', e)
        # client_socket.close()
        # cv2.destroyAllWindows()

    except BrokenPipeError as e:
        print('Exception: ', e)
        client_socket.close()
        # cv2.destroyAllWindows()

    except Exception as e:
        print('Exception: ', e)
        client_socket.close()

    # finally:
    #     cv2.destroyAllWindows()
    #     print('Socket closed')
