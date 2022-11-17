## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import sys

import logging
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
#import cv2

# Weight estimation
# sys.path.insert(1, '/data/lincoln/weight_estimation')
# import model_prediction_realsense
# from model_prediction_realsense import predict_weight

from py_inference import *
# from py_inference import inference

# Create a pipeline
pipeline = rs.pipeline()

#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# get intrinsic parameter
int_params = rs.intrinsics()

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
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

        outputs = predictor(color_image)
        outputs = outputs['instances'].to('cpu')
        boxes = outputs.pred_boxes if outputs.has("pred_boxes") else None
        # print(boxes)
        boxes = boxes.tensor.numpy()
        berries = np.zeros((boxes.shape[0], 3))

        # Check if x,y calculation is correct
        # cv2.imshow('RECEIVING FRAME', color_image)
        # for index, box in enumerate(boxes):
        #     x, y = int((box[2] + box[0]) / 2), int((box[3] + box[1]) / 2)
        #     z = depth_image[y, x]
        #     display_image = cv2.circle(color_image, (x, y), radius=50, color=(255, 0, 0))
        #     # berries[index] = [y, x, z]
        #     w_y, w_x, w_z = rs.rs2_deproject_pixel_to_point(int_params, [y, x], z/1000)
        #     berries[index] = [w_y, w_x, w_z]
        #     cv2.imshow('RECEIVING FRAME', display_image)

        # print(berries)
        # Detectron visualiser
        v = Visualizer(color_image[:, :, ::-1],
                       metadata=strawberry_metadata,
                       scale=0.5,
                       # remove the colors of unsegmented pixels.
                       # This option is only available for segmentation models
                       instance_mode=ColorMode.IMAGE_BW
                       )
        out = v.draw_instance_predictions(outputs)
        cv2.imshow('inference', out.get_image()[:, :, ::-1])

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            break


finally:
    pipeline.stop()
