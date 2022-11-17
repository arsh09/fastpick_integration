#!/usr/bin/env python
#! /usr/env/bin python


from __future__ import division
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import time
import subprocess, shlex, psutil

import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


# for forward kinematics
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics
from tf.transformations import quaternion_from_matrix
import tf


# 1.2952886754646897, 2.664927451405674, 34.47095671981777, 34.20000076293945
# 1.6197435616049916, 0.3429731077631004, 30.182617093191702,

CAMERA_Z = 30.182617093191702 / 100
CAMERA_Y = 0.3429731077631004 / 100
CAMERA_X =  1.6197435616049916 / 100


class WedgeToPalpateValidation: 

    def __init__(self): 

        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('berry_picking_automate', anonymous=True)
        self.robot = moveit_commander.RobotCommander()
        group_name = "panda_arm"
        self.group = moveit_commander.MoveGroupCommander(group_name)
        self.scene = moveit_commander.PlanningSceneInterface()
        self.display_trajectory_publisher = rospy.Publisher( "/move_group/display_planned_path", moveit_msgs.msg.DisplayTrajectory, queue_size=20)

        self.loop()


    def go_to_home(self, pose="ready"):
   
        rospy.loginfo ("Moving to home pose. Please wait...")
        self.group.set_max_velocity_scaling_factor(0.1)
        self.group.set_named_target(pose)
        self.group.go(wait=True)
        rospy.loginfo ("Moved to home pose...")
        time.sleep(1)

    
    def broadcast_and_transform(self, trans, rot, child, parent, new_parent):

        not_found = True
        listerner = tf.TransformListener()
        b = tf.TransformBroadcaster()
        
        while not_found:
            for i in range(10):
                b.sendTransform( trans, rot, rospy.Time.now(), child, parent )
                time.sleep(0.1)
                not_found = False
            try: 
                (trans1, rot1) = listerner.lookupTransform( new_parent, child, rospy.Time(0))
                rospy.loginfo("Translation: {} {} {}".format(trans1[0],trans1[1],trans1[2]))
                not_found = False
            except  : #( tf.lookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.loginfo("TF Exception")
                continue

        return trans1, rot1
 

    def move_robot_to_joint_trajectory(self, trans):
        
        # cartesian space path plan 
        waypoints = []

        self.group.set_max_velocity_scaling_factor(0.05)
        wpose = self.group.get_current_pose().pose
        wpose.position.x = trans[0]
        wpose.position.y = trans[1]
        wpose.position.z = trans[2]

        self.group.set_pose_target(wpose)
        self.group.go(wait=True)

        # waypoints.append( copy.deepcopy(wpose) )
        # # make cartesian path planning 
        # plan, fraction = self.group.compute_cartesian_path( waypoints, 0.01, 0.0 ) 

        # self.planned_path = self.group.retime_trajectory( self.robot.get_current_state(), plan, velocity_scaling_factor = 0.05) 

        # # display cartesian path 
        # display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        # display_trajectory.trajectory_start = self.robot.get_current_state()
        # display_trajectory.trajectory.append(self.planned_path)
        # # Publish
        # self.display_trajectory_publisher.publish(display_trajectory)

        
    def do_execute_planned_path(self):

        if (self.planned_path != None):
            self.group.execute(self.planned_path, wait=True)
 
    def loop(self):

        while not rospy.is_shutdown(): 
                # trans1 = [CAMERA_X, CAMERA_Y, CAMERA_Z]
                # # trans1 = [0,0,0.1]
                # rot1 = [0, 0, 0, 1]
                # trans, rot = self.broadcast_and_transform( trans1, rot1, 'berry', 'calibrated_frame', 'panda_link0')
            
            _input = raw_input("See awesoem output: ")

            if _input == 'h':
                self.go_to_home(pose="initial")
            elif _input == "c":
                trans1 = [CAMERA_X, CAMERA_Y, CAMERA_Z]
                # trans1 = [0,0,0.1]
                rot1 = [0, 0, 0, 1]
                trans, rot = self.broadcast_and_transform( trans1, rot1, 'berry', 'calibrated_frame', 'panda_link0')

                if len(trans) == 3:
                    self.move_robot_to_joint_trajectory(trans)
            

if __name__ == "__main__": 

    validation_exp = WedgeToPalpateValidation()
