#!/usr/bin/env python3

import time
from threading import Lock
from cv_bridge import CvBridge  # Package to convert between ROS and OpenCV Images
from std_msgs.msg import String
import rospy
from datetime import datetime
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import CameraInfo, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import sys
import os
import pyrealsense2 as rs
import threading
import sys
import numpy as np
import cv2
import os
import yaml
import sys
# from reconstruction_system.run_system import run_sys
import matplotlib.pyplot as plt
import matplotlib
from tools.rs_insertPointCloudSeg import MapManager

# import rosbag
# from pyntcloud import PyntCloud
matplotlib.use('TkAgg')
"""
ROS2 Package for Intel® RealSense™ Devices
http://www.lxshaw.com/tech/ros/2021/11/04/ros2realsense/
"""
# from examples.rs_insertPointCloud import callback, pointcloud_from_depth,rs_callback
# wkentaro
# /octomap-python
# https://github.com/wkentaro/octomap-python

# issue:
# https://github.com/mmatl/pyrender/issues/13

# realsense wrapper:
# https://github.com/IntelRealSense/librealsense/blob/development/wrappers/python/examples/opencv_pointcloud_viewer.py

# https://reposhub.com/python/miscellaneous/daavoo-pyntcloud.html
"""
The first one is for basic displaying of images using python.
detection blobs...........
"""
scaled_depth = 0.0
here = os.path.dirname(os.path.abspath(__file__))
# here = '/media/fpick/My Passport/Datasets/octomap/arc2017'
here = '/home/fpick/anaconda3/envs/env38/lib/python3.8/site-packages/imgviz/data/arc2017'
output_path = '/home/fpick/FPICK/Rlsense3DPrjTrack/results'
mutex = Lock()

# https://github.com/IntelRealSense/realsense-ros/issues/1342


class MsgManager(object):
    def __init__(self, topic):
        self.topic_color = topic
        self.bridge = CvBridge()
        self.topic_color = '/camera/color/image_raw'
        # '/camera/depth/image_rect_raw'
        self.topic_depth = '/camera/aligned_depth_to_color/image_raw'
        # '/camera/depth/image_rect_raw'
        self.topic_caminfo = '/camera/aligned_depth_to_color/camera_info'

        self.MapObj = MapManager()
        # self.intrinsics = None
        self.MapObj.depth_intrin = None  # self.intrinsics

        self.sub_color = rospy.Subscriber(
            self.topic_color, msg_Image, self.imageColorCallback)
        self.sub_depth = rospy.Subscriber(
            self.topic_depth, msg_Image, self.imageDepthCallback)
        self.sub_info = rospy.Subscriber(
            self.topic_caminfo, CameraInfo, self.imageDepthInfoCallback)

    def imageColorCallback(self, msg):
        # self.color_image = msg #msg.data
        self.MapObj.color_image = self.br.imgmsg_to_cv2(msg)
        # depth_image = np.asanyarray( MsgObj.depth_frame,dtype=np.uint8)
        # self.MapObj.color_image = np.asanyarray(self.MapObj.color_image,dtype=np.uint8)
        rospy.loginfo('color recieving from : %s', rospy.get_caller_id())
        # self.process_message()

    def imageDepthCallback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            self.MapObj.depth_frame = cv_image  # calculate the 3 d inoformation
            # Convert the depth image to a Numpy array
            # depth_array = np.array(self.MapObj.depth_image, dtype=np.float32)
            # he Depth scale is 1mm. ROS convention for uint16 depth image.

            pix = (data.width/2, data.height/2)
            sys.stdout.write('%s: Depth at center(%d, %d): %f(mm)\r' % (
                self.topic, pix[0], pix[1], cv_image[pix[1], pix[0]]))
            sys.stdout.flush()
            rospy.loginfo('depth recieving from : %s', rospy.get_caller_id())

            if self.MapObj.depth_intrin:
                depth = cv_image[pix[1], pix[0]]
                result = rs.rs2_deproject_pixel_to_point(
                    self.MapObj.depth_intrin, [pix[0], pix[1]], depth)
                rospy.loginfo('centrial point x y z :', result)
                # sys.stdout.write('%s: Depth at center(%d, %d): %f(mm)\r' % (self.topic, pix[0], pix[1], cv_image[pix[1], pix[0]]))
                # sys.stdout.flush()

        except CvBridgeError as e:
            print(e)
            return

    def imageDepthInfoCallback(self, cameraInfo):
        try:
            # import pdb; pdb.set_trace()
            if self.MapObj.depth_intrin:
                self.process_message()
                return
            self.MapObj.depth_intrin = rs.intrinsics()
            self.MapObj.depth_intrin.width = cameraInfo.width
            self.MapObj.depth_intrin.height = cameraInfo.height
            self.MapObj.depth_intrin.ppx = cameraInfo.K[2]
            self.MapObj.depth_intrin.ppy = cameraInfo.K[5]
            self.MapObj.depth_intrin.fx = cameraInfo.K[0]
            self.MapObj.depth_intrin.fy = cameraInfo.K[4]
            if cameraInfo.distortion_model == 'plumb_bob':
                self.MapObj.depth_intrin.model = rs.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.MapObj.depth_intrin.model = rs.distortion.kannala_brandt4
            self.MapObj.depth_intrin.coeffs = [i for i in cameraInfo.D]

        except CvBridgeError as e:
            print(e)
            return

        """    
        def depth_callback(self, msg):

        depth_image = self.br.imgmsg_to_cv2(msg)
        depth_image = depth_image.astype(np.float32)
        # self.MapObj.depth_image = self.MapObj.depth_image_oct.astype(np.uint8)
        depth_image[depth_image > 1.5] = 0.0
        depth_image[depth_image <= 0.0] = 'nan'
        # self.MapObj.depth_image[self.MapObj.depth_image > 0.5 ] = 0.0
        # self.MapObj.depth_image[self.MapObj.depth_image <= 0.0] = 'nan'
        self.MapObj.depth_image = depth_image
        # rospy.loginfo(rospy.get_caller_id() + 'depth recieving: %s', msg)
        rospy.loginfo('depth recieving from : %s', rospy.get_caller_id())"""

    """
    def caminfo_callback(self, msg):

        # lstinfo = np.array(msg)
        self.MapObj.camera_info = msg  # np.reshape(lstinfo,(3,3))
        # rospy.loginfo(rospy.get_caller_id() + 'cam info recieving: %s', msg)
        # self.caminfo_data.K[0, 0] = msg.K[0]  # fx
        # self.caminfo_data.K[1, 1] = msg.K[4]  # fy
        # self.caminfo_data.K[0, 2] = msg.K[2]  # ppx
        # self.caminfo_data.K[1, 2] = msg.K[5]  # ppy
        self.MapObj.width = int(self.MapObj.camera_info.width)
        self.MapObj.height = int(self.MapObj.camera_info.height)
        rospy.loginfo('cam info width :  %d', self.MapObj.width)
        rospy.loginfo('cam info height :  %d', self.MapObj.height)

        # self.MapObj.fx = self.MapObj.camera_info.K[0]
        # self.MapObj.fy = self.MapObj.camera_info.K[4]
        # self.MapObj.ppx = self.MapObj.camera_info.K[2]
        # self.MapObj.ppy = self.MapObj.camera_info.K[5]

        # self.MapObj.K[0, 0] = self.MapObj.fx
        # self.MapObj.K[1, 1] = self.MapObj.fy
        # self.MapObj.K[0, 2] = self.MapObj.ppx
        # self.MapObj.K[1, 2] = self.MapObj.ppy

        rospy.loginfo('cam info fx : %f', self.MapObj.K[0, 0])
        rospy.loginfo('cam info fy : %f', self.MapObj.K[1, 1])
        rospy.loginfo('cam info ppx : %f', self.MapObj.K[0, 2])
        rospy.loginfo('cam info ppy : %f', self.MapObj.K[1, 2])

        
        # mutex.acquire()
        self.process_message()
        # mutex.release()
    """

    def ShudownHook(self):
        print("shutdown time!")
        cv2.destroyAllWindows()
        exit()

    def process_message(self):
        # desired operation here
        # print('Edge is listening: depth-- {0}, color-- {1}, cam info-- {1}'.format(self.depth_image,self.color_image,self.caminfo_data))            cv2.imshow("camera", MsgObj.color_image)
        # Close down the video stream when done
        # if self.MapObj.flag==True:
        start_time = time.time()
        # rs_callback_octomap()
        # self.MapObj.centers, self.MapObj.depth_point = self.MapObj.rs_callback_cvSeg_bag()
        self.MapObj.centers, self.MapObj.depth_point = self.MapObj.rs_callback_cvSeg_cam()
        print("--- %s seconds ---" % (time.time() - start_time))
        # self.MapObj.flag=False

        # occupied, empty  = self.MapObj.rs_callback_o3d_rltime()
        self.MapObj.DetTrckPtPublisher()

        """  
        plt.figure('depth -rgb', figsize=(9, 6))
        plt.subplot ( 1, 2,1)
        plt.title ( 'Default params rgbd color image' )
        # im_rgb = cv2.cvtColor( np.asarray(rgbd_image.color), cv2.COLOR_BGR2RGB)
        # im_rgb = np.asarray(rgbd_image.color)
        # im_rgb = np.dstack((im_rgb,im_rgb,im_rgb)) #depth image is 1 channel, color is 3 channels
        # im_rgb = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2RGB)
        plt.imshow (MsgObj.depth_image)#,cmap='gray')
        plt.subplot ( 1 , 2 , 2 )
        plt.title ( 'Default params rgbd depth image' )
        plt.imshow (MsgObj.color_image)#,cmap='gray')
        # plt.subplot (2 , 2 , 3 )
        # plt.title ( 'Live Extracted rgbd  color image' )
        # im_rgb = cv2.cvtColor( np.asarray(rgbd.color), cv2.COLOR_BGR2RGB)
        # plt.imshow (im_rgb,cmap='gray')
        # plt.title ( 'Live Extracted regbd depth image' )
        # plt.subplot (2 , 2 , 4 )
        # plt.imshow (rgbd.depth)#,cmap='gray')
        plt.show (block=False)
        plt.pause(0.25)
        plt.close()
        """

        """ 
        scale_per = 0.5
        img = cv2.merge((self.MapObj.depth_image,self.MapObj.depth_image,self.MapObj.depth_image))
        img = img.astype(np.uint8)   
        
        img = np.hstack((img,self.MapObj.color_image)) 
        # img = self.MapObj.depth_image.astype(np.uint8)
        # img = cv2.merge(img,img, img)
        # img = np.hstack((img,self.MapObj.color_image)) 

        
        w = int(img.shape[1]*scale_per) 
        h = int(img.shape[0]*scale_per)  
        dim = (w,h)
        img = cv2.resize(img, dim,interpolation = cv2.INTER_AREA)      

        cv2.imshow("depth - color", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        """
        key = cv2.waitKey(100) & 0xFF   # Esc - kill, q, skipping
        if key == ord('q') or key == 27:
            # if key == 27:
            # cv2.destroyAllWindows()
            rospy.on_shutdown(self.ShudownHook)


class StreamManager(object):
    def __init__(self, pth='', img=''):

        self.webcamFlg = False  # choose webcam False: choose realsense camera
        self.rosbagFlg = True

        if self.webcamFlg == True:
            self.webcam_init()
        else:
            self.rs_init()

    def webcam_init(self):
        self.cap = cv2.VideoCapture()

    def rs_init(self):

        # Configure depth and color streams
        self.bagfile_path = '/home/fpick/Documents/data/20210827_121930.bag'
        self.rosbag_file = "/home/fpick/fpick_docker/out222.bag"
        self.output_path = '/home/fpick/FPICK/Results/'
        self.cam_info = CameraInfo()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.frmStart = 10
        self.frmCunt = -1
        # open rosbag file at start
        self.bag_in = rosbag.Bag(self.rosbag_file, "r")

        #########################################################
        self.topic_rbpose = ["/robot_pose"]
        self.messages_robpose = self.bag_in.read_messages(
            topics=self.topic_rbpose)  # Only get images data
        self.num_robpose = self.bag_in.get_message_count(
            topic_filters=self.topic_rbpose)
        print("# robot_pose topic:           " + self.topic_rbpose[0])
        print("# robot pose msg_count:       %u" % self.num_robpose)

        #########################################################

        self.topic_caminfo = ["/camera/camera2/color/camera_info"]
        self.messages_caminfo = self.bag_in.read_messages(
            topics=self.topic_caminfo)  # Only get images data
        self.num_caminfos = self.bag_in.get_message_count(
            topic_filters=self.topic_caminfo)
        print("# cam info topic:           " + self.topic_caminfo[0])
        print("# cam info msg_count:       %u" % self.num_caminfos)
        topic, self.msg_caminfo, t = next(self.messages_caminfo)

        print(self.msg_caminfo)
        D = self.msg_caminfo.D
        k = self.msg_caminfo.K
        P = self.msg_caminfo.P
        R = self.msg_caminfo.R

        print('fx= ', self.msg_caminfo.K[0])
        print('fy= ', self.msg_caminfo.K[4])
        print('ppx= ', self.msg_caminfo.K[2])
        print('ppy= ', self.msg_caminfo.K[5])

        #########################################################
        self.topic_img = ["/camera/camera2/color/image_raw"]
        self.messages_img = self.bag_in.read_messages(
            topics=self.topic_img)  # Only get images data
        self.num_images = self.bag_in.get_message_count(
            topic_filters=self.topic_img)
        print("# img topic:           " + self.topic_img[0])
        print("# img msg_count:       %u" % self.num_images)

        #########################################################
        self.topic_depth = ["/camera/camera2/aligned_depth_to_color/image_raw"]
        self.messages_alignedepth = self.bag_in.read_messages(
            topics=self.topic_depth)  # Only get images data
        self.num_alignedepth = self.bag_in.get_message_count(
            topic_filters=self.topic_depth)
        print("# aliged depth topic:           " + self.topic_depth[0])
        print("# aliged depth msg_count:       %u" % self.num_alignedepth)

        #########################################################
        self.topic_gps = ["/gps/filtered"]
        self.messages_gps = self.bag_in.read_messages(
            topics=self.topic_gps)  # Only get images data
        self.num_gps = self.bag_in.get_message_count(
            topic_filters=self.topic_gps)
        print("# gps topic:           " + self.topic_gps[0])
        print("# gps msg_count:       %u" % self.num_gps)
        ########################################################

        self.topicNames = {'/robot_pose',
                           '/camera/camera2/color/camera_info',
                           '/camera/camera2/color/image_raw',
                           '/camera/camera2/aligned_depth_to_color/image_raw',
                           '/gps/filtered'}

        np_poses = None
        keys_list = list(self.topicNames)
        a_key = keys_list[0]

        """
        ###################### realsense camera ##################
        # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_device_from_file(self.bagfile_path)

        # Start streaming
        self.profile = self.pipeline.start(self.config)

        self.depth_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.depth))
        self.depth_intrinsics = self.depth_profile.get_intrinsics()

        w, h = self.depth_intrinsics.width, self.depth_intrinsics.height
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        self.clipping_distance_in_meters = 1.0 # 1 meter 
        self.clipping_distance = self.clipping_distance_in_meters / self.depth_scale
        """
        """ 
        #深度图像向彩色对齐
        # align_to_color=rs.align(rs.stream.color)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        # Streaming loop
        for x in range(10):
            self.pipeline.wait_for_frames()
            self.frames = self.pipeline.wait_for_frames()
            self.aligned_frames = self.align.process(self.frames)
            self.depth_frame = self.aligned_frames.get_depth_frame()
            self.color_frame = self.aligned_frames.get_color_frame()

        # After alligned to color image  ---- Intrinsics & Extrinsics 
        self.depth_intrin_alligned = self.depth_frame.profile.as_video_stream_profile().intrinsics
        self.color_intrin_alligned = self.color_frame.profile.as_video_stream_profile().intrinsics
        """

        self.cam_info.K = np.array(self.cam_info.K).reshape(3, 3)
        # self.depth_intrin_alligned.height
        self.cam_info.height = self.msg_caminfo.height
        self.cam_info.width = self.msg_caminfo.width  # self.depth_intrin_alligned.width
        # self.cam_info.distortion_model =self.depth_intrin_alligned.model# params['distortion_model']
        # cam_info.K = params['camera_matrix']['data']
        # cam_info.D = params['distortion_coefficients']['data']
        # cam_info.R = params['rectification_matrix']['data']
        # cam_info.P = params['projection_matrix']['data']

        """

        K[0, 0] = msg_caninfo.K[0]#depth_intrin.fx
        K[1, 1] = msg_caninfo.K[4]#depth_intrin.fy
        K[0, 2] = msg_caninfo.K[2]#depth_intrin.ppx
        K[1, 2] = msg_caninfo.K[5]#depth_intrin.ppy
        """
        self.cam_info.K[0,
                        0] = self.msg_caminfo.K[0]  # depth_intrin_alligned.fx
        # depth_intrin_alligned.fy
        self.cam_info.K[1, 1] = self.msg_caminfo.K[4]
        # depth_intrin_alligned.ppx
        self.cam_info.K[0, 2] = self.msg_caminfo.K[2]
        # depth_intrin_alligned.ppy
        self.cam_info.K[1, 2] = self.msg_caminfo.K[5]
        self.cam_info.K = np.reshape(self.cam_info.K, 9)

    """

    # Streaming loop
    try:
        num = 0
        # .npz file reading and checking with numpy, python
        data = arc2017()
              
        # pems04_data = np.load('data/PEMS04/pems04.npz')
        # print(pems04_data.files) ##['data']
        # # go in and read the data set:
        # print(pems04_data['data'])#The output is an array
        # print(pems04_data['data'].shape)#(16992,307,3)
        while True:

            # occupied, empty  = callback(data)


            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Intrinsics & Extrinsics
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

            
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
            depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(
                color_frame.profile)


            depth_image = np.asanyarray(depth_frame.get_data(),dtype=np.uint8)
            color_image = np.asanyarray(color_frame.get_data(),dtype=np.uint8)

            # Stack both images horizontally
            src2 = np.zeros_like(color_image)
            src2[:,:,0] = depth_image
            src2[:,:,1] = depth_image
            src2[:,:,2] = depth_image
            # src2 = cv2.resize(depth_image, color_image.shape[1::-1])
            images = np.hstack((src2,color_image))

            figname='strw_octomap_depth_rgb_'+str(num)+'.png'
            num = num+1
            nampepath = os.path.join(output_path, figname) 
            # images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
            cv2.imwrite(nampepath,cv2.cvtColor(images, cv2.COLOR_BGR2RGB))
            fig = plt.figure('Demon', figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
            plt.title('Rs Depth & Colors')
            plt.imshow(images,cmap='gray')
            plt.show(block=True)


            # Convert images to numpy arrays
            depth_image = 0.001*np.asanyarray(depth_frame.get_data(),dtype=np.float32)
            color_image = np.asanyarray(color_frame.get_data(),dtype=np.uint8)
            ############################
            ############################


            # res = np.hstack((color_image,frame)) #stacking images side-by-side
            # display with meshing purpose
            # res_mesh = np.hstack((cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB), cv2.cvtColor(depth_colormap_aligned, cv2.COLOR_BGR2RGB)))      
            # res_mesh = np.hstack((cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB), depth_colormap_aligned))      

            # display with tracking purose
            # res_track = np.hstack((cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            # figname='strw_mesh_'+str(num)+'.png'
            # nampepath = os.path.join(output_path, figname) 
            # res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            # cv2.imwrite(nampepath,cv2.cvtColor(res_mesh, cv2.COLOR_BGR2RGB))

            camera_info = data['camera_info']
            camera_info['width'] = 1280
            camera_info['height'] = 720
            dim = (camera_info['width'], camera_info['height'])
            # depth_image= ~depth_image
            # resize image
            depth_image = cv2.resize(depth_image, dim, interpolation = cv2.INTER_AREA)
            color_image = cv2.resize(color_image, dim, interpolation = cv2.INTER_AREA)
            
            
            K = np.array(camera_info['K']).reshape(3, 3)
            tempD=data['depth']
            depth_image[depth_image > 0.75 ] = 0.0
            depth_image[depth_image <= 0.0] = 'nan'
            data['depth'] = depth_image
            data['rgb'] = color_image
            K[0, 0] = depth_intrin.fx
            K[1, 1] = depth_intrin.fy
            K[0, 2] = depth_intrin.ppx
            K[1, 2] = depth_intrin.ppy
            ##############################
            ##############################

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            scaled_depth=cv2.convertScaleAbs(depth_image, alpha=0.08)
            depth_colormap = cv2.applyColorMap(scaled_depth, cv2.COLORMAP_JET)

            
            
            
            # pc = rs.pointcloud()
            # pc.map_to(color_frame)
            # pointcloud = pc.calculate(depth_frame)
            # pointcloud.export_to_ply("11.ply", color_frame)
            # cloud = PyntCloud.from_file("11.ply")
            # # pcd = np.array(cloud.xyz)
            # pcd = np.array(pointcloud.data)
            # pcd = np.array(pointcloud.xyz)

            # # newarr = pcd.reshape(720,1280,3)
            # tmp = np.array([pcd.tolist()])
            # newarr = tmp.reshape(720,1280,3)
            # centroids = cloud.centroid
            # occupied, empty  = rs_callback(pcd,data)
            # cloud.plot()
            # cv2.imshow('RealSense', cloud)
            StreamManager
            # cv2.imshow('RealSense', occupied)
            # cv2.imshow('RealSense', empty)

            
            truncated_depth=thresholdDepth(scaled_depth)
            truncated_depth=detectBlobs(truncated_depth)
            cv2.imshow('Truncated Depth', truncated_depth)
            
            
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
    finally:
        # pipeline.stop()
        cv2.destroyAllWindows()

    """
