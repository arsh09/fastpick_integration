#!/usr/bin/env python

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose2D, Point
from math import pi
import subprocess, shlex, psutil
from cv_bridge import CvBridge
import cv2 
import numpy as np
import tf
import pyrealsense2
from sensor_msgs.msg import PointCloud2
from uph_msgs.msg import DetectedBerries

import os
import time
import json

from ethercat_interface.msg import Cmd


# predict weight (because why not) 
# from joblib import dump, load
# MODEL_FILE_MULTI = '/home/robofruit/uph_franka_ws/src/franka_uph_project/uph_simulation/uph_control/models/multi_distance.joblib'
# MODEL_FILE_SINGLE = '/home/robofruit/uph_franka_ws/src/franka_uph_project/uph_simulation/uph_control/models/single_berry_single_distance.joblib'
# regr_multi = load(MODEL_FILE_MULTI)
# regr_single = load(MODEL_FILE_SINGLE)

class DataCollection: 

    def __init__(self):     
        rospy.init_node('automatic_berry_picking_node')

        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        group_name = "panda_arm"
        self.group = moveit_commander.MoveGroupCommander(group_name)
        self.scene = moveit_commander.PlanningSceneInterface()
        self.display_trajectory_publisher = rospy.Publisher( "/move_group/display_planned_path", moveit_msgs.msg.DisplayTrajectory, queue_size=20)
        self.planned_path = None
        self.berry_translation_from_base = []

        self.gripper_pub = rospy.Publisher('/pos_cmd', Cmd, queue_size=10)

        self.detected_subs = rospy.Subscriber('/detected_berry', DetectedBerries, self.detected_berries_cb)
        self.detected_berries_msg = None
        self.detected_berries_msg_received = False

        self.detected_subs_bottom = rospy.Subscriber('/detected_berry_bottom', DetectedBerries, self.detected_berries_bottom_cb)
        self.detected_berries_msg_bottom = None
        self.detected_berries_msg_received_bottom = False

        self.bridge = CvBridge()
        self.img_sub = rospy.Subscriber('/aligned_depth_to_color/camera_info', CameraInfo, self.color_camera_info_cb)
        self.img_sub = rospy.Subscriber('/color/image_raw', Image, self.color_image_cb)
        self.depth_sub = rospy.Subscriber('/aligned_depth_to_color/image_raw', Image, self.depth_image_cb)
        self.img_msg = None
        self.depth_msg = None
        self.img_info_msg = None
        self.img_msg_received = False
        self.depth_msg_received = False
        self.img_info_msg_received = False
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.select_point_cb)
        
        self.pre_graph_pose = None
        self.robot_speed_factor = 0.075
        self.pre_grasp_distance = 0.15

        berry_correction_topic = rospy.Subscriber('/detected_berry_correction', Pose2D, self.berry_correction_cb)
        self.berry_in_base_pub = rospy.Publisher('/berry_in_base', Point, queue_size = 10 )

        self.visual_xy_berry = []
        self.do_visual = False
        self.amazing_loop = 0

        self.sample_count = 0
        self.loop()

    def berry_correction_cb(self, msg) :
        x, y = msg.x, msg.y 
        self.visual_xy_berry = [x,y]
        error_y = 20
        error_x = 20

    def detected_berries_cb(self, msg):
        self.detected_berries_msg = msg
        self.detected_berries_msg_received = True
        
    def detected_berries_bottom_cb(self, msg):
        self.detected_berries_msg_bottom = msg
        self.detected_berries_msg_received_bottom = True
        
    def select_point_cb(self, event,x,y,flags,param):
        if self.img_msg_received and self.img_info_msg_received and self.depth_msg_received:

            if event == cv2.EVENT_LBUTTONDOWN: # Left mouse button
                rospy.loginfo("Mouse event: {} {}".format(x,y))
                self.pointX = x
                self.pointY = y

    def convert_2d_to_3d(self,x , y, depth):
  
        _intrinsics = pyrealsense2.intrinsics()
        _intrinsics.width = self.img_info_msg.width
        _intrinsics.height = self.img_info_msg.height
        _intrinsics.ppx = self.img_info_msg.K[2]
        _intrinsics.ppy = self.img_info_msg.K[5]
        _intrinsics.fx = self.img_info_msg.K[0]
        _intrinsics.fy = self.img_info_msg.K[4]
        _intrinsics.model  = pyrealsense2.distortion.none  
        _intrinsics.coeffs = [i for i in self.img_info_msg.D]  
  
        result = pyrealsense2.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth )  
        # rospy.loginfo("3D point: {} {} {}".format( result[0]/1000, result[1]/1000, result[2]/1000))
        trans = [result[0]/1000 , result[1]/1000, result[2]/1000 ]
        rot = [0, 0, 0 , 1]
        
        self.broadcast_and_transform(trans, rot, "/top_camera_color_optical_frame" , "/berry_point" , "/panda_link0")

    def broadcast_and_transform(self, trans, rot, parent, child, new_parent):

        not_found = True
        listerner = tf.TransformListener()
        b = tf.TransformBroadcaster()
        while not_found:
            for i in range(10): 
                b.sendTransform( trans, rot, rospy.Time.now(), child, parent )
                time.sleep(0.1)
                
            try: 
                (trans1, rot1) = listerner.lookupTransform( new_parent, child, rospy.Time(0))
                self.berry_translation_from_base = trans1
                
                # self.berry_translation_from_base[0] -= self.pre_grasp_distance
                # self.berry_translation_from_base[1] += 0.0300
                # self.berry_translation_from_base[2] += 0.01

                # self.berry_translation_from_base[0] -= 0.03750
                # self.berry_translation_from_base[1] -= self.pre_grasp_distance
                # self.berry_translation_from_base[2] += 0.0125

                not_found = False
            except ( tf.lookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.loginfo("TF Exception")
                continue

    def depth_image_cb(self, data):
        self.depth_msg_received = True
        self.depth_msg = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        
    def color_camera_info_cb(self, data):
        self.img_info_msg_received = True
        self.img_info_msg = data

    def color_image_cb(self , data):        
        self.img_msg_received = True
        self.img_msg = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        self.img_msg = cv2.cvtColor(self.img_msg, cv2.COLOR_BGR2RGB)
    
    def move_to_graph_pose_align(self):
          
        # cartesian space path plan 
        waypoints = []
        wpose = self.group.get_current_pose().pose
        if (len(self.berry_translation_from_base) == 3 ):
            x1, y1, z1 = self.berry_translation_from_base 
            x2, y2, z2 = wpose.position.x,  wpose.position.y, wpose.position.z  
            self.pre_graph_pose = [wpose.position.x,  wpose.position.y, wpose.position.z]

            x2 = x1
            # y2 = y1

            count = 10
            dx, dy, dz = (x1 - x2)/count, (y1-y2)/count, (z1-z2)/count

            for i in range(count):
                
                wpose.position.x += dx
                wpose.position.y += dy
                wpose.position.z += dz

                waypoints.append( copy.deepcopy(wpose) )
        
            # make cartesian path planning 
            plan, fraction = self.group.compute_cartesian_path( waypoints, 0.01, 0.0 ) 
            print "Cartesian plan fraction is {}".format(fraction)
            self.planned_path = self.group.retime_trajectory( self.robot.get_current_state(), plan, velocity_scaling_factor = self.robot_speed_factor) 

            # display cartesian path 
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = self.robot.get_current_state()
            display_trajectory.trajectory.append(self.planned_path)
            # Publish
            self.display_trajectory_publisher.publish(display_trajectory)

    def move_to_pre_grasp_pose(self):
          
        # cartesian space path plan 
        waypoints = []
        wpose = self.group.get_current_pose().pose
        if (len(self.berry_translation_from_base) == 3 ):
            x1, y1, z1 = self.berry_translation_from_base
            x1 -= self.pre_grasp_distance
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
            print "Cartesian plan fraction is {}".format(fraction)
            self.planned_path = self.group.retime_trajectory( self.robot.get_current_state(), plan, velocity_scaling_factor = self.robot_speed_factor) 

            # display cartesian path 
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = self.robot.get_current_state()
            display_trajectory.trajectory.append(self.planned_path)
            # Publish
            self.display_trajectory_publisher.publish(display_trajectory)

    def move_to_graph_pose(self):
          
        # cartesian space path plan 
        waypoints = []
        wpose = self.group.get_current_pose().pose
        if (len(self.berry_translation_from_base) == 3 ):
            x1, y1, z1 = wpose.position.x,  wpose.position.y, wpose.position.z 
            x1 += ( self.pre_grasp_distance )
            x1 -= 0.02
            # z1 -= 0.02
            # x1 += ( self.pre_grasp_distance )
            # x1, y1, z1 = wpose.position.x,  wpose.position.y + self.pre_grasp_distance + 0.01, wpose.position.z 
            x2, y2, z2 = wpose.position.x,  wpose.position.y, wpose.position.z  
            
            count = 3
            dx, dy, dz = (x1 - x2)/count, (y1-y2)/count, (z1-z2)/count

            for i in range(count):
                
                wpose.position.x += dx
                wpose.position.y += dy
                wpose.position.z += dz

                waypoints.append( copy.deepcopy(wpose) )
        
            # make cartesian path planning 
            plan, fraction = self.group.compute_cartesian_path( waypoints, 0.01, 0.0 ) 
            print "Cartesian plan fraction is {}".format(fraction)
            self.planned_path = self.group.retime_trajectory( self.robot.get_current_state(), plan, velocity_scaling_factor = self.robot_speed_factor) 

            # display cartesian path 
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = self.robot.get_current_state()
            display_trajectory.trajectory.append(self.planned_path)
            # Publish
            self.display_trajectory_publisher.publish(display_trajectory)

    def move_to_post_graph_pose(self):
          
        # cartesian space path plan 
        waypoints = []
        wpose = self.group.get_current_pose().pose
        if (len(self.berry_translation_from_base) == 3 ):
            x1, y1, z1 = self.pre_graph_pose[0], self.pre_graph_pose[1], self.pre_graph_pose[2]
            x2, y2, z2 = wpose.position.x,  wpose.position.y, wpose.position.z  
            # x2 = x1

            count = 10
            dx, dy, dz = (x1 - x2)/count, (y1-y2)/count, (z1-z2)/count

            for i in range(count):
                
                wpose.position.x += dx
                wpose.position.y += dy
                wpose.position.z += dz

                waypoints.append( copy.deepcopy(wpose) )
        
            # make cartesian path planning 
            plan, fraction = self.group.compute_cartesian_path( waypoints, 0.01, 0.0 ) 
            print "Cartesian plan fraction is {}".format(fraction)
            self.planned_path = self.group.retime_trajectory( self.robot.get_current_state(), plan, velocity_scaling_factor = self.robot_speed_factor) 

            # display cartesian path 
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = self.robot.get_current_state()
            display_trajectory.trajectory.append(self.planned_path)
            # Publish
            self.display_trajectory_publisher.publish(display_trajectory)

    def do_execute_planned_path(self):

        if (self.planned_path != None):
            self.group.execute(self.planned_path, wait=False)

    def go_to_home(self, pose="ready"):
        print "Moving to {} pose. Please wait...".format(pose)
        self.group.set_max_velocity_scaling_factor(self.robot_speed_factor)
        self.group.set_named_target(pose)
        self.group.go(wait=False)
        print "Moved to {} pose...".format(pose)
        time.sleep(1)

    def open_gripper(self):
        cmd  = Cmd()
        cmd.position1 = 0
        cmd.position2 = 0
        cmd.velocity1 = 100
        cmd.velocity2 = 100
        for i in range (10):
            self.gripper_pub.publish(cmd)
        # time.sleep(5)

    def close_gripper(self):
        cmd  = Cmd()
        cmd.position1 = 55000
        cmd.position2 = 0
        cmd.velocity1 = 100
        cmd.velocity2 = 100
        for i in range(10):
            self.gripper_pub.publish(cmd)
        # time.sleep(5)

    def open_cutter(self):
        cmd  = Cmd()
        cmd.position1 = 0
        cmd.position2 = 2500
        cmd.velocity1 = 100
        cmd.velocity2 = 100
        for i in range(10):
            self.gripper_pub.publish(cmd)
    
        # time.sleep(5)

    def close_cutter(self):
        cmd  = Cmd()
        cmd.position1 = 0
        cmd.position2 = 250
        cmd.velocity1 = 100
        cmd.velocity2 = 100
        for i in range(10):
            self.gripper_pub.publish(cmd)
        # time.sleep(5)

    def plot_berry_frame(self):
        
        if self.detected_berries_msg_received:
            max_score_index= 0
            scores = [berry.score for berry in self.detected_berries_msg.detected_berries]
            masks = [berry.masks for berry in self.detected_berries_msg.detected_berries]
            bboxes = [berry.bbox for berry in self.detected_berries_msg.detected_berries]
            keypoints = [np.array( berry.keypoints ).reshape((5,3)) for berry in self.detected_berries_msg.detected_berries]
            all_x = []
            all_y = []
            all_z = []
            all_pick_point = []
            actual_mask = np.zeros( self.img_msg.shape[:2], dtype=np.uint8)

            for count in range(len(scores)):
                mask_indices = np.asarray(masks[count]).reshape(2,-1)
                indexes = tuple( map(tuple, mask_indices) )
                mask = actual_mask.copy()
                mask[indexes] = 255
                
                depth_valid_index = np.where(mask > 0)
                depth_values = self.depth_msg[depth_valid_index]
                
                y, x = np.median( np.array(depth_valid_index)[0,:] ), np.median( np.array(depth_valid_index)[1,:] )
                depth = np.median( depth_values )

                all_pick_point.append( keypoints[count][0, :] )
                # print ( x, y, keypoints[count][0,:] )
                x1, y1 = bboxes[count][0], bboxes[count][1]
                x2, y2 = bboxes[count][2], bboxes[count][3]

                if x1 < self.pointX and x2 > self.pointX and y1 < self.pointY and y2 > self.pointY:
                    max_score_index = count
                    

                all_x.append(x)
                all_y.append(y)
                all_z.append(depth)

            

            x,y, z= all_pick_point[max_score_index][0], all_pick_point[max_score_index][1], all_z[max_score_index]
            x -= 22.5

            self.convert_2d_to_3d(x,y,z)
            
    def find_berry_orientation(self, keypoint):

        x1, y1, c1 = keypoint[0,:]
        x2, y2, c2 = keypoint[1,:]
        x3, y3 =  x2, y1
        min_confidence = 0.01
        angle = 0
        if c1 > min_confidence and c2 > min_confidence:
            angle = np.arctan2 ( (x1 - x2) , (y2 - y1) )

    def plot_rect_mask(self):
        image_copy = self.img_msg.copy()
        try:
            if self.detected_berries_msg_received:
                pluckables = [berry.pluckable for berry in self.detected_berries_msg.detected_berries]
                bboxes = [berry.bbox for berry in self.detected_berries_msg.detected_berries]
                masks = [berry.masks for berry in self.detected_berries_msg.detected_berries]
                keypoints = [berry.keypoints for berry in self.detected_berries_msg.detected_berries]
                scores = [berry.score for berry in self.detected_berries_msg.detected_berries]

                actual_mask = np.zeros( self.img_msg.shape[:2], dtype=np.uint8)

                for count, bbox in enumerate(bboxes):
                    x1, y1 = int(bbox[0]) , int(bbox[1])
                    x2, y2 = int(bbox[2]) , int(bbox[3])
                    
                    # mask_indices = np.asarray(masks[count]).reshape(2,-1)
                    # indexes = tuple( map(tuple, mask_indices) )
                    # mask = actual_mask.copy()
                    # mask[indexes] = 255

                    # visMask =  mask[y1:y2, x1:x2]                
                    # roi = image_copy[y1:y2, x1:x2]
                    # colors = np.ones( roi.shape )
                    
                    # if pluckables[count] == 0:
                    #     colors[:,:,1] = 255
                    # else:
                    #     colors[:,:,2] = 255

                    # roi = cv2.bitwise_and( roi, roi , mask = visMask )
                    # blended = ( roi * 0.8 + 0.2*colors  ).astype("uint8")

                    # image_copy[y1:y2, x1:x2] = blended
                    cv2.rectangle( image_copy, (x1,y1), (x2,y2), (0,255,0), 1) 

                    # keypoint = keypoints[count]
                    # keypoint = np.array(keypoint).reshape((5,3))
                    # total_kp_confidence = 0
                    # for i in range(1):
                    #     x,y,c = keypoint[i, :]
                    #     x,y = int(x), int(y)
                    #     cv2.circle( image_copy, (x,y), 5, colors, -1)
                    #     total_kp_confidence += c

                    # text = "{}: {:.3f}  {}: {:.3f}".format("KP", total_kp_confidence, "Score", scores[count])
                    # cv2.putText( image_copy, text, (x1-10, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colors[0,0,:], 1)
        except: 
            pass

        w, h, c = image_copy.shape
        factor = 1
        w, h = int( w/factor ) , int( h/factor)
        image_copy = cv2.resize( image_copy, (h,w))
        return image_copy


    def move_up(self):
        self.group.set_max_velocity_scaling_factor(0.01)
        desired_pose = self.group.get_current_pose()
        desired_pose.pose.position.z += 0.02
        self.group.set_pose_target( desired_pose ) 
        self.group.go(wait=False)

    def move_down(self):
        self.group.set_max_velocity_scaling_factor(0.01)
        desired_pose = self.group.get_current_pose()
        desired_pose.pose.position.z -= 0.02
        self.group.set_pose_target( desired_pose ) 
        self.group.go(wait=False)

    def move_left(self):
        self.group.set_max_velocity_scaling_factor(0.01)
        desired_pose = self.group.get_current_pose()
        desired_pose.pose.position.y += 0.01
        self.group.set_pose_target( desired_pose ) 
        self.group.go(wait=False)

    def move_right(self):
        self.group.set_max_velocity_scaling_factor(0.01)
        desired_pose = self.group.get_current_pose()
        desired_pose.pose.position.y -= 0.01
        self.group.set_pose_target( desired_pose ) 
        self.group.go(wait=False)

    def visual_servo(self):

        self.do_visual = True
        self.amazing_loop = 0
        # while self.do_visual:
        #     pass

    def collect_data(self):        
        
        if self.detected_berries_msg_received_bottom:
            pluckables = [berry.pluckable for berry in self.detected_berries_msg_bottom.detected_berries]
            bboxes = [berry.bbox for berry in self.detected_berries_msg_bottom.detected_berries]
            keypoints = [berry.keypoints for berry in self.detected_berries_msg_bottom.detected_berries]

            data = {
                "pluckabes" : pluckables,
                "bboxes" : bboxes,
                "keypoints" : keypoints
            }
            
            file_paths = "/home/robofruit/Desktop/data/sample_" + str(self.sample_count) + ".json"
            self.sample_count += 1
            with open(file_paths, "w") as f:
                json.dump(data, f, indent = 4) 
        
    def loop(self):

        while not rospy.is_shutdown(): 

            if self.img_msg_received and self.depth_msg_received:
                image_copy = self.plot_rect_mask()
                
                cv2.imshow('image', image_copy)
                
                k = cv2.waitKey(33)

                if k == 27:   
                    break
                elif k==-1: 
                    continue
                else:
                    if k == ord('n'):
                        self.collect_data()

                    if k == ord('d'):
                        self.plot_berry_frame()

                    if k == ord('g'):
                        self.move_to_graph_pose_align()
                        print "Planning for pre-grasp-align pose. Press 'e' to execute the shown cartesian plan"

                    if k == ord('f'):
                        self.move_to_pre_grasp_pose()
                        print "Planning for pre-graph-pose. Press 'e' to execute the shown cartesian plan"

                    if k == ord('j'):
                        self.move_to_graph_pose()
                        print ("Planning for graph pose. Press 'e' to execute the shown cartesian plan") 

                    if k == ord('l'):
                        self.move_to_post_graph_pose()
                        print "Planning for post-grasp pose. Press 'e' to execute the shown cartesian plan"

                    if k == ord('e'):
                        self.do_execute_planned_path()

                    if k == ord('v'):
                        print "Visual servoing-based correction"
                        self.visual_servo()

                    if k == ord('h'):
                        self.go_to_home(pose="initial")

                    if k == ord('k'):
                        self.go_to_home(pose="Basket-B")

                    if k == ord('1'):
                        print ("Opening gripper...")
                        self.open_gripper()

                    if k == ord('2'):
                        print ("Closing gripper...")
                        self.close_gripper()

                    if k == ord('3'):
                        print ("Opening separator. It will open the gripper as well")
                        self.open_cutter()

                    if k == ord('4'):
                        print ("Closing separator. It will open the gripper as well")
                        self.close_cutter()

                    if k == 81: # left 
                        self.move_left()
                    if k == 83: # right
                        self.move_right()
                    if k == 82: # up
                        self.move_up()
                    if k == 84: # down
                        self.move_down()

                    if k == ord('5'):
                        print ("Automatic state machine for different pose execution")
                        for i in range(1):
                            self.go_to_home(pose="initial")
                            print ("Closing separator...")
                            self.close_cutter()
                            self.go_to_home(pose="initial1")
                            print ("Opening separator...")
                            self.open_cutter()
                            print ("Closing gripper...")
                            self.close_gripper()
                            self.go_to_home(pose="initial")
                            self.go_to_home(pose="place")
                            print ("Opening gripper...")
                            self.open_gripper()
                            self.go_to_home(pose="initial")


                    if k == ord('u'):
                        print ("Automatic state machine for pick and place berry")
                        for i in range(1):
                            self.go_to_home(pose="initial")

                            # print "Closing separator..."
                            # self.close_cutter()

                            print "Planning for pre-grasp-align pose."
                            self.move_to_graph_pose_align() 
                            self.do_execute_planned_path()

                            print "Planning for pre-grasp-pose."
                            self.move_to_pre_grasp_pose()
                            self.do_execute_planned_path()
    
                            print "Opening separator..."
                            self.open_cutter()
                            
                            print "Visual servoing-based correction"
                            time.sleep(1)
                            self.visual_servo()

                            print "Planning for grasp pose."
                            self.move_to_graph_pose()
                            self.do_execute_planned_path()

                            print ("Closing gripper...")
                            self.close_gripper()

                            # print "Opening separator..."
                            # self.open_cutter()

                            # print "Opening gripper..."
                            # self.open_gripper()

                            print "Planning for post-grasp."
                            self.move_to_post_graph_pose()
                            self.do_execute_planned_path()

                            self.go_to_home(pose="initial")

                            # self.go_to_home(pose="Basket-A")

                            # print ("Opening gripper...")
                            # self.open_gripper()

                            # self.go_to_home(pose="initial")


        cv2.destroyAllWindows()

if __name__ == "__main__": 

    data_collection = DataCollection()
