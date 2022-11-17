#!/usr/bin/env python

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from sensor_msgs.msg import Image, CameraInfo
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
import time
import subprocess, shlex, psutil
from cv_bridge import CvBridge
import cv2 
import numpy as np
import tf
import pyrealsense2
# Switch controller server client libs
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest , SwitchControllerResponse
from controller_manager_msgs.srv import ListControllers, ListControllersRequest , ListControllersResponse
from sensor_msgs.msg import PointCloud2

# for point cloud processing 
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
import tf2_py as tf2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import ros_numpy

from ethercat_interface.msg import Cmd

class DataCollection: 

    def __init__(self): 

        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('reach_to_berry_node', anonymous=True)
        self.robot = moveit_commander.RobotCommander()

        group_name = "panda_arm"
        self.group = moveit_commander.MoveGroupCommander(group_name)
        self.scene = moveit_commander.PlanningSceneInterface()
        self.display_trajectory_publisher = rospy.Publisher( "/move_group/display_planned_path", moveit_msgs.msg.DisplayTrajectory, queue_size=20)

        self.bridge = CvBridge()

        self.gripper_pub = rospy.Publisher('/pos_cmd', Cmd, queue_size=10)

        self.selected_points = rospy.Subscriber('/rviz_selected_points', PointCloud2, self.selected_points_cb)
        self.img_sub = rospy.Subscriber('/color/camera_info', CameraInfo, self.color_camera_info_cb)
        self.img_sub = rospy.Subscriber('/color/image_raw', Image, self.color_image_cb)
        self.depth_sub = rospy.Subscriber('/depth/image_rect_raw', Image, self.depth_image_cb)
        self.img_msg = None
        self.depth_msg = None
        self.img_info_msg = None
        self.img_msg_received = False
        self.depth_msg_received = False
        self.img_info_msg_received = False
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.select_point_cb)
 
        # recording rosbag proces 
        self.rosbag_proc = None
        self.is_recording = False

        self.pointX = 0
        self.pointY = 0
        self.region = 5
        self.translation = []
        self.pointcloud = None
        self.pointcloud_msg = None

        self.planned_path = None

        self.loop() 

    def process_selected_points (self):

        target_frame = "panda_link0"
        timeout = 0.0
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.logwarn("Please wait 1 seconds for the tf")
        time.sleep(1)
        while True : 
            try:
                trans = self.tf_buffer.lookup_transform(target_frame, self.pointcloud_msg.header.frame_id, rospy.Time())  
                cloud_out = do_transform_cloud(self.pointcloud_msg, trans)
                self.pointcloud = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(cloud_out)
                break
            except tf2.LookupException as ex:
                rospy.logwarn(ex)
                break
            except tf2.ExtrapolationException as ex:
                rospy.logwarn(ex)
                break

        trans = [ np.mean(self.pointcloud[:, 0]) , np.mean(self.pointcloud[:, 1]) + 0.0275, np.mean(self.pointcloud[:, 2]) ]
        rot = [0, 0, 0, 1]
        rospy.loginfo(trans)
        self.broadcast_and_transform(trans, rot, "panda_link0" , "/berry_point" , "/panda_link0")
        self.translation = trans
        

    def selected_points_cb(self, msg):
        msg.header.frame_id = "top_camera_depth_optical_frame"
        self.pointcloud_msg = msg
        self.process_selected_points()


    def select_point_cb(self, event,x,y,flags,param):
        if self.img_msg_received and self.img_info_msg_received and self.depth_msg_received:

            if event == cv2.EVENT_LBUTTONDOWN: # Left mouse button
                rospy.loginfo("Mouse event: {} {}".format(x,y))
                self.pointX = x
                self.pointY = y


    def median_of_depth(self, x, y):

        region = self.region # pixels for depth
        if self.depth_msg_received:
            points = self.depth_msg[ y - region : y + region , x - region : x + region ]
            
            depth_mean = np.mean( points )
            depth_median = np.median( points )
            depth_min = np.min(points)
            depth_max = np.max(points)
            depth_value = self.depth_msg[y,x] 
            print ("Depth: Median: {}  Mean: {}  Exact: {}   Minimum: {}   Maximum: {}".format(depth_median, depth_mean, depth_value, depth_min, depth_max))
            print ("depth median is : ", depth_mean, depth_median, self.depth_msg[y,x] )

        return depth_median


    def convert_2d_to_3d(self):
  
        x, y = self.pointX, self.pointY
        _intrinsics = pyrealsense2.intrinsics()
        _intrinsics.width = self.img_info_msg.width
        _intrinsics.height = self.img_info_msg.height
        _intrinsics.ppx = self.img_info_msg.K[2]
        _intrinsics.ppy = self.img_info_msg.K[5]
        _intrinsics.fx = self.img_info_msg.K[0]
        _intrinsics.fy = self.img_info_msg.K[4]
        _intrinsics.model  = pyrealsense2.distortion.none  
        _intrinsics.coeffs = [i for i in self.img_info_msg.D]  
  
        self.median_of_depth(x,y)
        result = pyrealsense2.rs2_deproject_pixel_to_point(_intrinsics, [x, y], self.median_of_depth(x,y))  

        rospy.loginfo("3D point: {} {} {}".format( result[0]/1000, result[1]/1000, result[2]/1000))

        trans = [result[0]/1000, result[1]/1000, result[2]/1000 ]
        rot = [0, 0, 0 , 1]
        
        self.broadcast_and_transform(trans, rot, "top_camera_color_optical_frame" , "/berry_point" , "/panda_link0")

        
    def broadcast_and_transform(self, trans, rot, parent, child, new_parent):

        not_found = True
        listerner = tf.TransformListener()
        b = tf.TransformBroadcaster()
        while not_found:
            for i in range(10): 
                b.sendTransform( trans, rot, rospy.Time.now(), child, parent )
                time.sleep(0.1)
            
            break

            # try: 
            #     (trans1, rot1) = listerner.lookupTransform( new_parent, child, rospy.Time(0))
            #     rospy.loginfo("Translation: {} {} {}".format(trans1[0],trans1[1],trans1[2]))
            #     self.translation = trans1
            #     not_found = False
            # except ( tf.lookupException, tf.ConnectivityException, tf.ExtrapolationException):
            #     rospy.loginfo("TF Exception")
            #     continue

        # self.record_bag_file()
        # self.plan_with_camera(trans1)

    def depth_image_cb(self, data):
        self.depth_msg_received = True
        # The depth image is a single-channel float32 image
        # the values is the distance in mm in z axis
        self.depth_msg = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        
    def color_camera_info_cb(self, data):
        self.img_info_msg_received = True
        self.img_info_msg = data

    def color_image_cb(self , data):
        
        self.img_msg_received = True
        self.img_msg = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')

    def move_robot_to_joint_trajectory(self):
          
        # cartesian space path plan 
        waypoints = []
        wpose = self.group.get_current_pose().pose
        
        if (len(self.translation) == 3 ):
            x1, y1, z1 = self.translation 
            x2, y2, z2 = wpose.position.x,  wpose.position.y, wpose.position.z  
            count = 10

            dx, dy, dz = (x1 - x2)/count, (y1-y2)/count, (z1-z2)/count

            for i in range(count):
                
                wpose.position.x += dx
                wpose.position.y += dy
                wpose.position.z += dz

                waypoints.append( copy.deepcopy(wpose) )
        
            # make cartesian path planning 
            plan, fraction = self.group.compute_cartesian_path( waypoints, 0.01, 0.0 ) 
            print ("Plan fraction is {}".format(fraction))
            self.planned_path = self.group.retime_trajectory( self.robot.get_current_state(), plan, velocity_scaling_factor = 0.15) 

            # display cartesian path 
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = self.robot.get_current_state()
            display_trajectory.trajectory.append(self.planned_path)
            # Publish
            self.display_trajectory_publisher.publish(display_trajectory)

    def do_execute_planned_path(self):

        if (self.planned_path != None):
            self.group.execute(self.planned_path, wait=True)



    def plan_with_camera(self):

        self.group.set_max_velocity_scaling_factor(0.05)
        current_pose = self.group.get_current_pose().pose
        rospy.loginfo("{} {}".format(self.group.get_end_effector_link(), self.group.get_planning_frame()))
        pose_goal = geometry_msgs.msg.PoseStamped()
        pose_goal.header.stamp = rospy.Time.now()
        pose_goal.header.frame_id = "panda_link0"       
        pose_goal.pose.orientation = current_pose.orientation
        
        if (len(self.translation) == 3 ):
            pose_goal.pose.position.x = self.translation[0]
            pose_goal.pose.position.y = self.translation[1]
            pose_goal.pose.position.z = self.translation[2]

            self.group.set_pose_target(pose_goal)
            self.group.go(wait=True)

    def go_to_home(self, pose="ready"):
        rospy.loginfo ("Moving to home pose. Please wait...")
        self.group.set_max_velocity_scaling_factor(0.1)
        self.group.set_named_target(pose)
        self.group.go(wait=True)
        rospy.loginfo ("Moved to home pose...")
        time.sleep(1)

    def close_gripper(self):

        cmd  = Cmd()
        cmd.position1 = 0
        cmd.position2 = 0
        cmd.velocity1 = 100
        cmd.velocity2 = 100

        for i in range (10):
            self.gripper_pub.publish(cmd)
    
        time.sleep(3)


    def open_gripper(self):

        cmd  = Cmd()
        cmd.position1 = 55000
        cmd.position2 = 0
        cmd.velocity1 = 100
        cmd.velocity2 = 100
        for i in range(10):
            self.gripper_pub.publish(cmd)
    
        time.sleep(3)

    def open_cutter(self):

        cmd  = Cmd()
        cmd.position1 = 0
        cmd.position2 = 2500
        cmd.velocity1 = 100
        cmd.velocity2 = 100
        for i in range(10):
            self.gripper_pub.publish(cmd)
    
        time.sleep(3)

    def close_cutter(self):

        cmd  = Cmd()
        cmd.position1 = 0
        cmd.position2 = 0
        cmd.velocity1 = 100
        cmd.velocity2 = 100
        for i in range(10):
            self.gripper_pub.publish(cmd)
    
        time.sleep(3)

    def loop(self):

        while not rospy.is_shutdown(): 
            # pass
            if self.img_msg_received and self.depth_msg_received:

                cv2.rectangle( self.img_msg, (self.pointX - self.region , self.pointY - self.region), (self.pointX + self.region , self.pointY + self.region) ,  (255,0,0), 2)
                cv2.imshow('image', self.img_msg)

                k = cv2.waitKey(33)
                if k==27:    # Esc key to stop
                    break
                elif k==-1:  # normally -1 returned,so don't print it
                    continue
                else:
                    if k == ord('h'):
                        rospy.loginfo("Should go to home pose")
                        self.go_to_home(pose='initial')

                    if k == ord('p'):
                        rospy.loginfo("Should go to place pose")
                        self.go_to_home(pose='place')

                    if k == ord('s'):
                        # self.convert_2d_to_3d()
                        self.process_selected_points()

                    if k == ord('d'):
                        self.move_robot_to_joint_trajectory()
                        
                    if k == ord('e'):
                        self.do_execute_planned_path()
                    
                    if k == ord('o'):##
                        rospy.loginfo("Opening gripper")
                        self.open_gripper()
                    
                    if k == ord('c'):
                        rospy.loginfo("Closing gripper")
                        self.close_gripper()
                    
                    if k == ord('v'):
                        rospy.loginfo("Opening separator")
                        self.open_cutter()

                    if k == ord('b'):
                        rospy.loginfo("Closing separator")
                        self.close_cutter()

        cv2.destroyAllWindows()

if __name__ == "__main__": 

    data_collection = DataCollection()
