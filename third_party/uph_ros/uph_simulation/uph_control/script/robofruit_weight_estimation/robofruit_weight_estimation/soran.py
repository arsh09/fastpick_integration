## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
from datetime import datetime

import pyrealsense2.pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
#import cv2
from pyntcloud import PyntCloud
import os
import sys

# from file_var import *
# from python_tkinter import key, myClick, callback, get_point, callback_rect, rs_call

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

# point cloud
pc = rs.pointcloud()

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
clipping_distance_in_meters = 0.20  # 20 cms
clipping_distance = clipping_distance_in_meters / depth_scale

group_id = 'prediction'

dataset_dir = '/home/robofruit/localdrive/myprojects/camera_calib/'
time_stamp = 1
# latency = 10
# counter = 1
# Streaming loop
try:
    while True:

        # while group_id == 0:
        #     group_id = input('group_id: ')
        #     print('group_id: ', group_id)
        #     canvas_directory = dataset_dir + '/' + group_id
        #     if os.path.isdir(canvas_directory):
        #         print('Folder with group_id exists: ', group_id)
        #         group_id = 0
        #     else:
        #         os.mkdir(canvas_directory)
        # if not (counter % latency) == 0:
        #     counter +=1
        #     print('counter: ', counter)
        #     continue
        # counter = 1
        # Get frameset of color and depth

        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = aligned_depth_frame.profile.get_extrinsics_to(color_frame.profile)

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        pc.map_to(color_frame)
        pointcloud_3 = pc.calculate(aligned_depth_frame)

        # if image_id == 3:
        #     clipping_distance_in_meters = 0.25  # 1 meter
        #     clipping_distance = clipping_distance_in_meters / depth_scale

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack(
            (depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # coloriser
        colourised_depth = np.asarray(rs.colorizer().colorize(aligned_depth_frame).get_data())
        # print(colourised_depth.shape)

        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # images = np.hstack((bg_removed, colourised_depth, color_image))
        images = np.hstack((bg_removed, colourised_depth, color_image))
        # cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)

        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            # cv2.destroyAllWindows()
            break
            # pipeline.stop()
            # sys.exit(0)


        # s for save
        # elif key == 115:

        to_panda = predict_weight(color_image, depth_image)

        if len(to_panda) == 0:
            continue

        publish = []
        for index, data in enumerate(to_panda):

            box, avg_depth = data[0], data[-1]

            x, y = int((box[2] + box[0]) / 2), int((box[3] + box[1]) / 2)

            center_depth = aligned_depth_frame.get_distance(int(x), int(y))

            depth_point_2 = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth_scale)
            color_point_2 = rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point_2)

            print('colour_point_x, colour_point_x', color_point_2[0] * 10**4, color_point_2[1] * 10**4)

            display_image = cv2.circle(color_image, (x, y), radius=5, color=(255, 0, 0))

            collect_data = (color_point_2[0] * 10**4, color_point_2[1] * 10**4, avg_depth, center_depth * 100)

            # time_stamp = datetime.now().strftime("%m_%h_%y_%H_%M_%S")

            cv2.imshow('Berry center', display_image)
            key = cv2.waitKey(1)
            if key == 32:

                if not os.path.isdir(dataset_dir + '/' + str(time_stamp)):
                    os.makedirs(dataset_dir + '/' + str(time_stamp))

                else:
                    raise IsADirectoryError

                np.save(dataset_dir + '/' + str(time_stamp) + '/depth_' + str(time_stamp), depth_image)
                cv2.imwrite(dataset_dir + '/' + str(time_stamp) + '/' + str(time_stamp) + '.png', color_image)
                np.save(dataset_dir + '/' + str(time_stamp) + '/points_' + str(time_stamp), collect_data)
                print('data saved')
                time_stamp += 1

            publish.append(collect_data)

            # Press esc or 'q' to close the image window
            # if key & 0xFF == ord('q') or key == 27:
            #     cv2.destroyAllWindows()
            #     break
            #     raise KeyboardInterrupt

        print(publish)


except KeyboardInterrupt as e:
    print(e)
    # pipeline.stop()

finally:
    cv2.destroyAllWindows()
    pipeline.stop()

# sys.path.append(os.getcwd())
# os.system('python ./python_tkinter.py ' + canvas_directory)