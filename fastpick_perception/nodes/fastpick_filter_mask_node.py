#!/usr/bin/env python3

'''
- This node subscribes (color, depth, camera info and detected berry)
- Draw the rectangle around detected berries in color image
- Filter depth image into two images foreground (i.e. berry depth) and bg (everything else).
- Then it re-publishes the two depth, color and camera info.
- The republished depths are used for pointcloud 

Muhammad Arshad 
02/3/2022
'''

from __future__ import print_function
from six.moves import input

import sys
import rospy 
import moveit_commander
from sensor_msgs.msg import Image, CameraInfo
from fastpick_msgs.msg import StrawberryObject, StrawberryObjects
from geometry_msgs.msg import TransformStamped, PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf as tf1_ros 
import tf2_ros 
import pyrealsense2

class MaskToDepthFilter:

    def __init__(self):

        moveit_commander.roscpp_initialize(sys.argv)
        self.scene = moveit_commander.PlanningSceneInterface()

        # get topic names 
        color_topic_ = rospy.get_param("~color_topic_", "/color/image_raw")
        depth_topic_ = rospy.get_param("~depth_topic_", "/aligned_depth_to_color/image_raw")
        caminfo_topic_ = rospy.get_param("~caminfo_topic_", "/aligned_depth_to_color/camera_info")
        berry_topic_ = rospy.get_param("~berry_topic", "/detected_berries")

        re_color_topic_ = rospy.get_param("~re_color_topic_", "/fastpick_perception/color")
        re_depth_fg_topic_ = rospy.get_param("~re_depth_fg_topic_", "/fastpick_perception/depth_berry")
        re_depth_bg_topic_ = rospy.get_param("~re_depth_bg_topic_", "/fastpick_perception/depth_background")
        re_caminfo_topic_ = rospy.get_param("~re_caminfo_topic_", "/fastpick_perception/camera_info")

        self.depth_stats = rospy.get_param("~depth_stats", "median") 

        # republish the masked depth and color (because why not)
        self.filtered_color_pub = rospy.Publisher( re_color_topic_, Image, queue_size = 5)
        self.filtered_depth_fg_pub = rospy.Publisher( re_depth_fg_topic_, Image, queue_size = 5)
        self.filtered_depth_bg_pub = rospy.Publisher( re_depth_bg_topic_, Image, queue_size = 5)
        self.filtered_caminfo_pub = rospy.Publisher( re_caminfo_topic_, CameraInfo, queue_size = 5)

        # perception topics
        self.depth_frame = ""
        self.color_encoding = ""
        self.depth_encoding = ""
        self.rgb_img = None
        self.depth_img = None
        self.camera_params = None
        self.bridge = CvBridge()

        self.rgb_img_subs = rospy.Subscriber( color_topic_, Image, self.rgb_img_cb)
        self.depth_img_subs = rospy.Subscriber( depth_topic_, Image, self.depth_img_cb)
        self.camera_info_subs = rospy.Subscriber( caminfo_topic_, CameraInfo, self.camera_info_cb)
 
        # detection topics 
        self.detected_berries = []
        self.detected_berries_subs = rospy.Subscriber(berry_topic_, StrawberryObjects, self.detected_berries_cb)


        self.berries_in_scene = []

        self.base_to_cam_transform = TransformStamped()
        self.base_to_grasp_link_transform = TransformStamped()
        self.rate = rospy.Rate(1)

        self.handle_loop()

    def test_object(self): 
        rospy.loginfo( self.scene.get_known_object_names() )
        berry_pose = PoseStamped()
        berry_pose.header.frame_id = "panda_link0"
        berry_pose.header.stamp = rospy.Time.now()
        berry_pose.pose.position.x = 0.5
        berry_pose.pose.position.y = 0.0
        berry_pose.pose.position.z = 0.58
        
        berry_pose.pose.orientation.x = 0.0 
        berry_pose.pose.orientation.y = 0.0 
        berry_pose.pose.orientation.z = 0.0 
        berry_pose.pose.orientation.w = 1.0 

        self.scene.add_sphere( "test" , berry_pose, radius = 0.015 )

    def rgb_img_cb(self, data):
        self.color_encoding = data.encoding
        self.rgb_img = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
    
    def depth_img_cb(self, data):
        self.depth_encoding = data.encoding
        self.depth_frame = data.header.frame_id
        self.depth_img = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        
    def camera_info_cb(self, data ):
        self.camera_params = data

    def detected_berries_cb(self, data):
        self.detected_berries = data

    def convert_berry_to_frame(self, x, y, depth, child_frame_id ): 

        berry_frame_br = tf2_ros.TransformBroadcaster()
        intrinsics_param = pyrealsense2.intrinsics()
        intrinsics_param.width = self.camera_params.width
        intrinsics_param.height = self.camera_params.height
        intrinsics_param.ppx = self.camera_params.K[2]
        intrinsics_param.ppy = self.camera_params.K[5]
        intrinsics_param.fx = self.camera_params.K[0]
        intrinsics_param.fy = self.camera_params.K[4]
        intrinsics_param.model  = pyrealsense2.distortion.none  
        intrinsics_param.coeffs = [i for i in self.camera_params.D]  
  
        translation = pyrealsense2.rs2_deproject_pixel_to_point(intrinsics_param, [x, y], depth )  
        translation = [ x / 1000 for x in translation ]

        if abs( translation[2] ) > 0.005 : 
            if self.base_to_cam_transform.header.frame_id == "":
                rotation = [0.0, 0.0, 0.0 , 1.0]
            else: 
                rot = self.base_to_cam_transform.transform.rotation
                rotation = [ rot.x, rot.y, rot.z, rot.w ]

            rotation = [0.0, 0.0, 0.0 , 1.0]

            _t = self.base_to_cam_transform.transform.translation
            _r = self.base_to_cam_transform.transform.rotation
            base_to_cam_translation = tf1_ros.transformations.translation_matrix( [ _t.x, _t.y, _t.z ] )
            base_to_cam_rotation = tf1_ros.transformations.quaternion_matrix( [ _r.x, _r.y, _r.z, _r.w ] )
            base_to_cam_tf = np.dot( base_to_cam_translation, base_to_cam_rotation )

            cam_to_berry_translation = tf1_ros.transformations.translation_matrix( [ translation[0], translation[1], translation[2] ] )
            cam_to_berry_rotation = tf1_ros.transformations.quaternion_matrix( [ 0.0, 0.0, 0.0, 1.0 ] )
            cam_to_berry_tf = np.dot( cam_to_berry_translation, cam_to_berry_rotation )

            base_to_berry_tf = np.dot( base_to_cam_tf, cam_to_berry_tf )
            base_to_berry_translation = tf1_ros.transformations.translation_from_matrix(base_to_berry_tf)
            base_to_berry_rotation = tf1_ros.transformations.quaternion_from_matrix(base_to_berry_tf)
            
            _berry_transform = TransformStamped()
            _berry_transform.header.stamp = rospy.Time.now()
            _berry_transform.header.frame_id = self.base_to_cam_transform.header.frame_id
            _berry_transform.child_frame_id = child_frame_id 
            _berry_transform.transform.translation.x = base_to_berry_translation[0]
            _berry_transform.transform.translation.y = base_to_berry_translation[1]
            _berry_transform.transform.translation.z = base_to_berry_translation[2] 
            _berry_transform.transform.rotation.x = self.base_to_grasp_link_transform.transform.rotation.x
            _berry_transform.transform.rotation.y = self.base_to_grasp_link_transform.transform.rotation.y
            _berry_transform.transform.rotation.z = self.base_to_grasp_link_transform.transform.rotation.z
            _berry_transform.transform.rotation.w = self.base_to_grasp_link_transform.transform.rotation.w
    
            berry_frame_br.sendTransform( _berry_transform )

        # self.add_berry_in_moveit_scene( _berry_transform )

    def find_base_to_camera_rotation(self, base_frame, camera_frame): 
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)
        transform = TransformStamped()
        not_found = True
        count = 0
        while not_found: 
            try: 
                transform = tf_buffer.lookup_transform(base_frame, camera_frame, rospy.Time())
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                count += 1
                self.rate.sleep()
                if (count > 10): 
                    break

        return transform

    def add_berry_in_moveit_scene(self, _berry_transform ): 
        
        berry_name = _berry_transform.child_frame_id
        self.berries_in_scene = self.scene.get_known_object_names()
        
        berry_pose = PoseStamped()
        berry_pose.header.frame_id = "panda_link0"
        berry_pose.header.stamp = rospy.Time.now()
        berry_pose.pose.position.x = _berry_transform.transform.translation.x
        berry_pose.pose.position.y = _berry_transform.transform.translation.y
        berry_pose.pose.position.z = _berry_transform.transform.translation.z

        berry_pose.pose.orientation.x = self.base_to_grasp_link_transform.transform.rotation.x
        berry_pose.pose.orientation.y = self.base_to_grasp_link_transform.transform.rotation.y
        berry_pose.pose.orientation.z = self.base_to_grasp_link_transform.transform.rotation.z
        berry_pose.pose.orientation.w = self.base_to_grasp_link_transform.transform.rotation.w

        self.scene.add_sphere( berry_name , berry_pose, radius = 0.015 )

 
    def draw_bbox_on_image(self, color, depth):

        self.base_to_cam_transform = self.find_base_to_camera_rotation( "panda_link0", "top_camera_color_optical_frame")
        self.base_to_grasp_link_transform = self.find_base_to_camera_rotation( "panda_link0", "fastpick_grasp_link")

        full_mask = np.zeros( color.shape[:2] , dtype=np.uint8)
        if not ( isinstance( self.detected_berries, type([]) )):
            for berry in self.detected_berries.berries: 

                # draw rectangle
                x1, y1, x2, y2 = berry.bbox
                cv2.rectangle( color, (x1,y1), (x2, y2), (255,255,255) , 1)

                # recreate the mask on ros side.
                indices = (np.array(berry.mask)).reshape( (2, int( len(berry.mask)/2) ) )
                indices = tuple(map(tuple, indices))
                full_mask[indices] = 255

                # for each berry, find the depth value (in mm) 
                # and the x,y in image plane
                _mask_ = np.zeros( color.shape[:2] , dtype=np.uint8)
                _mask_[indices] = 255
                _depth_values = depth[ _mask_ == 255 ] 
                xc , yc = (x1 + x2)/2 , (y1+y2)/2 
                if self.depth_stats.lower() == "mean" : 
                    depth_value = np.mean( _depth_values )
                else: 
                    depth_value = np.median( _depth_values )

                self.convert_berry_to_frame( xc, yc, depth_value, "berry_{}".format(berry.id))
            
            cv2.imshow("full Mask: ", full_mask)
            depth_bg = depth.copy() 
            zero_indices = np.where( full_mask == 0 )
            depth[zero_indices] = 0

            # rospy.loginfo( "Min: {}Max: {}Mean: Median: {}".format( np.min( depth_bg ), np.max( depth_bg ) , np.mean( depth_bg ) , np.median( depth_bg ) )) 

            # backgroun depth filter (because why not :D)
            min_depth , max_depth = 500, 1250 # mm
            indices_near = np.where( depth_bg < min_depth ) 
            indices_far = np.where( depth_bg > max_depth ) 
            depth_bg[indices_near] = 0
            depth_bg[indices_far] = 0

            zero_indices = np.where( full_mask != 0 )
            depth_bg[zero_indices] = 0

            if not ( isinstance( self.camera_params, type(None)) ):
                # convert color and depth to ROS image
                color_ros = self.bridge.cv2_to_imgmsg(color, encoding=self.color_encoding)
                depth_ros = self.bridge.cv2_to_imgmsg(depth, encoding=self.depth_encoding)
                depth_bg_ros = self.bridge.cv2_to_imgmsg(depth_bg, encoding=self.depth_encoding)
                
                color_ros.header.frame_id = self.depth_frame
                depth_ros.header.frame_id = self.depth_frame
                depth_bg_ros.header.frame_id = self.depth_frame

                color_ros.header.stamp = rospy.Time().now()
                depth_ros.header.stamp = color_ros.header.stamp
                depth_bg_ros.header.stamp = color_ros.header.stamp
                self.camera_params.header.stamp = color_ros.header.stamp

                self.filtered_color_pub.publish( color_ros ) 
                self.filtered_depth_fg_pub.publish( depth_ros ) 
                self.filtered_depth_bg_pub.publish( depth_bg_ros )
                self.filtered_caminfo_pub.publish( self.camera_params )

        return color, depth, full_mask 
 
    def handle_exit(self):
        cv2.destroyAllWindows()
        self.rgb_img_subs.unregister()
        self.depth_img_subs.unregister()
        self.camera_info_subs.unregister()
        rospy.loginfo("Gracefully exiting...")

    def handle_loop(self):
        
        rospy.loginfo("Starting depth mask filering node.")
        
        while not rospy.is_shutdown():
            try:
                if not isinstance( self.rgb_img, type(None) )  and ( not isinstance( self.depth_img, type(None) ) ):
                    color = self.rgb_img.copy()
                    depth = self.depth_img.copy()  
                    color, depth, mask = self.draw_bbox_on_image(color, depth)
                    k = cv2.waitKey(1)
                    if k == 27: 
                        self.handle_exit()
                        break

            except KeyboardInterrupt:
                self.handle_exit()
                break
        

if __name__ == '__main__' : 

    rospy.init_node('fastpick_perception_filter_mask_node')
    maskToDepthFilter = MaskToDepthFilter()