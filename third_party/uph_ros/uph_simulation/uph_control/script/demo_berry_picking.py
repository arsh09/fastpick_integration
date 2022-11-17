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


from franka_control.msg import ErrorRecoveryAction, ErrorRecoveryActionGoal

import os
import time
import json

from ethercat_interface.msg import Cmd


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

        # tracking
        self.trackers = cv2.MultiTracker_create()
        self.is_tracker_box_added = False

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
        cv2.namedWindow('TopCamera')
        cv2.setMouseCallback('TopCamera', self.select_point_cb)
        
        self.pre_graph_pose = None
        self.robot_speed_factor = 0.075
        self.pre_grasp_distance = 0.075

        berry_correction_topic = rospy.Subscriber('/detected_berry_correction', Pose2D, self.berry_correction_cb)
        self.berry_in_base_pub = rospy.Publisher('/berry_in_base', Point, queue_size = 10 )

        self.franka_recovery_topic = rospy.Publisher('/franka_control/error_recovery/goal', ErrorRecoveryActionGoal, queue_size=10)


        self.cutterPosition = -1
        self.grippoerPosition = -1
        
        self.gripper_movement(gripper_position=0, cutter_position = 0)
        self.loop()

    def berry_correction_cb(self, msg) :
        x, y = msg.x, msg.y 
        self.visual_xy_berry = [x,y]
        error_y = 20
        error_x = 20

    def detected_berries_cb(self, msg):
        self.detected_berries_msg = msg
        self.detected_berries_msg_received = True
        self.handle_tracker_object()
        
    def detected_berries_bottom_cb(self, msg):
        self.detected_berries_msg_bottom = msg
        self.detected_berries_msg_received_bottom = True
        
    def select_point_cb(self, event,x,y,flags,param):
        if self.img_msg_received and self.img_info_msg_received and self.depth_msg_received:

            if event == cv2.EVENT_LBUTTONDOWN: # Left mouse button
                # rospy.loginfo("Mouse event: {} {}".format(x,y))
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
                # self.berry_translation_from_base[1] += 0.0100
                self.berry_translation_from_base[2] += 0.02

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

        roll, pitch, yaw = 0, 0.15, 0        
        quaternion = tf.transformations.quaternion_from_euler( roll, pitch, yaw )
        if (len(self.berry_translation_from_base) == 3 ):
            x1, y1, z1 = self.berry_translation_from_base 
            x2, y2, z2 = wpose.position.x,  wpose.position.y, wpose.position.z  
            self.pre_graph_pose = [wpose.position.x,  wpose.position.y, wpose.position.z]
            x2 = x1

            count = 10
            dx, dy, dz = (x1 - x2)/count, (y1-y2)/count, (z1-z2)/count
            for i in range(count):
                wpose.position.x += dx
                wpose.position.y += dy
                wpose.position.z += dz
                wpose.orientation.x = quaternion[0]
                wpose.orientation.y = quaternion[1]
                wpose.orientation.z = quaternion[2]
                wpose.orientation.w = quaternion[3]

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
            # y1 += 0.01
            x2, y2, z2 = wpose.position.x,  wpose.position.y, wpose.position.z  
            count = 5
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
            x1 += ( self.pre_grasp_distance - 0.025 )
            # z1 += 0.02
            x2, y2, z2 = wpose.position.x,  wpose.position.y, wpose.position.z  
            
            count = 5
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

    def gripper_movement(self, gripper_position = 0, cutter_position = 0):

        if gripper_position >= 0 and gripper_position <= 55000 and cutter_position >= 0 and cutter_position <= 2500:
            cmd  = Cmd()
            cmd.position1 = gripper_position
            cmd.position2 = cutter_position
            cmd.velocity1 = 100
            cmd.velocity2 = 100
            for i in range (10):
                self.gripper_pub.publish(cmd)

            self.cutterPosition = cutter_position
            self.grippoerPosition = gripper_position
        

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

                x1, y1 = bboxes[count][0], bboxes[count][1]
                x2, y2 = bboxes[count][2], bboxes[count][3]
                # center of bbox (on x-axis)
                xc, yc = int ((x1+x2)/2), int( (y1))

                all_pick_point.append( [xc,yc, 1] )

                # picking point (first key-point)
                # all_pick_point.append( keypoints[count][0, :] )
                
                # if x1 < self.pointX and x2 > self.pointX and y1 < self.pointY and y2 > self.pointY:
                #     max_score_index = count
                    
                all_x.append(x)
                all_y.append(y)
                all_z.append(depth)

            
            try:
                x,y, z= all_pick_point[max_score_index][0], all_pick_point[max_score_index][1], all_z[max_score_index]
                x -= 22.5
                self.convert_2d_to_3d(x,y,z)

            except:
                print ("Detection changed.")
            
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
                    

                    mask_indices = np.asarray(masks[count]).reshape(2,-1)
                    indexes = tuple( map(tuple, mask_indices) )
                    mask = actual_mask.copy()
                    mask[indexes] = 255

                    visMask =  mask[y1:y2, x1:x2]                
                    roi = image_copy[y1:y2, x1:x2]
                    colors = np.ones( roi.shape )
                    
                    if pluckables[count] == 0:
                        colors[:,:,1] = 255
                    else:
                        colors[:,:,2] = 255

                    roi = cv2.bitwise_and( roi, roi , mask = visMask )
                    blended = ( roi * 0.9 + 0.1*colors  ).astype("uint8")

                    image_copy[y1:y2, x1:x2] = blended
                    cv2.rectangle( image_copy, (x1,y1), (x2,y2), (0,255,0), 1) 

                    keypoint = keypoints[count]
                    keypoint = np.array(keypoint).reshape((5,3))
                    total_kp_confidence = 0
                    for i in range(1):
                        x,y,c = keypoint[i, :]
                        x,y = int(x), int(y)
                        cv2.circle( image_copy, (x,y), 5, (0,255,0), -1)
                        total_kp_confidence += c

                    xc, yc = int ((x1+x2)/2), int( (y1))
                    cv2.circle( image_copy, (xc,yc), 5, (0,255,255), -1)

                    text = "{}: {:.3f}".format("KP", total_kp_confidence)
                    cv2.putText( image_copy, text, (x1-5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
                    text = "{}: {:.3f}".format("Score", scores[count])
                    cv2.putText( image_copy, text, (x1-5, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

        except: 
            print ("plot rect mask error")
            pass

        w, h, c = image_copy.shape
        factor = 1
        w, h = int( w/factor ) , int( h/factor)
        image_copy = cv2.resize( image_copy, (h,w))
        return image_copy


    def handle_tracker_object(self):
        if not self.is_tracker_box_added:
            for i in range(self.detected_berries_msg.total):
                bbox = self.detected_berries_msg.detected_berries[i].bbox
                x1, y1, x2, y2 = bbox
                x, y, w, h = int(x1), int (y1), int(x2 - x1), int (y2 - y1)
                box = (x,y,w,h)
                tracker = cv2.TrackerKCF_create()
                tracker = cv2.TrackerTLD_create()
                self.trackers.add( tracker, self.img_msg, box )
            self.is_tracker_box_added = True

    def handle_tracker_update(self):
        image_copy = self.img_msg.copy()
        (success, boxes) = self.trackers.update(image_copy)
        if success:
            for count, box in enumerate(boxes): 
                (x,y,w,h) = [int(v) for v in box]
                cv2.rectangle(image_copy, (x,y), (x+w, y+h), (0,255,0), 2)
                text = "id:{}".format(count+1)
                cv2.putText( image_copy, text, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        else:
            self.is_tracker_box_added = success

        return image_copy
    
    def franka_error_recovery(self):

        print ("Recoving franka control")
        msg = ErrorRecoveryActionGoal()
        for i in range(10):
            self.franka_recovery_topic.publish(msg)
            time.sleep(0.01)

    def loop(self):

        while not rospy.is_shutdown(): 

            if self.img_msg_received and self.depth_msg_received:
                image_copy1 = self.plot_rect_mask()
                # image_copy2 = self.handle_tracker_update()
                # image_copy = np.hstack((image_copy1, image_copy2))
                
                # w, h = image_copy.shape[:2]
                # factor = 1
                # w, h = int( w/factor ) , int( h/factor)
                # image_copy = cv2.resize( image_copy, (h,w))
                cv2.imshow('TopCamera', image_copy1)
                
                k = cv2.waitKey(33)

                if k == 27:   
                    break
                elif k==-1: 
                    continue
                else: 
                    # print (k)       
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
 
                    if k == ord('h'):
                        self.go_to_home(pose="initial")

                    if k == ord('k'):
                        self.go_to_home(pose="Basket-B")

                    if k == ord('1'):
                        print ("Opening gripper...")
                        self.gripper_movement(gripper_position = 0, cutter_position = self.cutterPosition)

                    if k == ord('2'):
                        print ("Closing gripper...")
                        self.gripper_movement(gripper_position = 55000, cutter_position = self.cutterPosition)

                    if k == ord('3'):
                        print ("Opening cutter...")
                        self.gripper_movement(gripper_position = self.grippoerPosition, cutter_position = 2500)

                    if k == ord('4'):
                        print ("Closing cutter...")
                        self.gripper_movement(gripper_position = self.grippoerPosition, cutter_position = 100)


                    if k == 44: # <
                        self.cutterPosition -= 250
                        self.gripper_movement(gripper_position = self.grippoerPosition, cutter_position = self.cutterPosition)

                    if k == 46: # >
                        self.cutterPosition += 250
                        self.gripper_movement(gripper_position = self.grippoerPosition, cutter_position = self.cutterPosition)
  

                    if k == 190:
                        self.go_to_home(pose="place1")
                    if k == 191:
                        self.go_to_home(pose="place2")
                    if k == 192:
                        self.go_to_home(pose="place3")
                    if k == 193:
                        self.go_to_home(pose="place4")
                    # if k == 194:
                    #     self.go_to_home(pose="punnet5")
                    # if k == 195:
                    #     self.go_to_home(pose="punnet6")
   

                    if k == ord('r'): 
                        self.franka_error_recovery()


                    if k == ord('q'): 
                        desired_pose = self.group.get_current_pose()
                        desired_pose.pose.position.z += 0.01
                        self.group.set_pose_target( desired_pose ) 
                        self.group.go(wait=True)

                    if k == ord('a'): 
                        desired_pose = self.group.get_current_pose()
                        desired_pose.pose.position.z -= 0.01
                        self.group.set_pose_target( desired_pose ) 
                        self.group.go(wait=True)

                    if k == ord('w'): 
                        desired_pose = self.group.get_current_pose()
                        desired_pose.pose.position.y += 0.01
                        self.group.set_pose_target( desired_pose ) 
                        self.group.go(wait=True)

                    if k == ord('s'): 
                        desired_pose = self.group.get_current_pose()
                        desired_pose.pose.position.y -= 0.01
                        self.group.set_pose_target( desired_pose ) 
                        self.group.go(wait=True)

                    if k == 93: # ]
                        desired_pose = self.group.get_current_pose()
                        desired_pose.pose.position.x += 0.01
                        self.group.set_pose_target( desired_pose ) 
                        self.group.go(wait=True)

                    if k == 91: # [
                        desired_pose = self.group.get_current_pose()
                        desired_pose.pose.position.x -= 0.01
                        self.group.set_pose_target( desired_pose ) 
                        self.group.go(wait=True)


        cv2.destroyAllWindows()

if __name__ == "__main__": 

    data_collection = DataCollection()
