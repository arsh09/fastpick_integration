#!/usr/bin/env python3

'''
Muhammad Arshad 
03/03/2022
'''
import sys
import rospy
import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import Pose2D, Point, PoseStamped, TransformStamped
import numpy as np
import tf2_ros
import time 

class FastPickActionServer: 

    def __init__(self): 

            
        rospy.init_node("fastpick_action_server_node")
    
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        group_name = "panda_arm"
        self.group = moveit_commander.MoveGroupCommander(group_name)
        self.scene = moveit_commander.PlanningSceneInterface()
        self.display_trajectory_publisher = rospy.Publisher( "/move_group/display_planned_path", moveit_msgs.msg.DisplayTrajectory, queue_size=20)
        self.planned_path = None

        self.rate = rospy.Rate(1)

        self.child_frame = rospy.get_param("~berry_frame", "berry_1")
        self.parent_frame = "panda_link0"

        self.offset_z  = -0.05
        berry_transform = self.get_berry_frame( self.parent_frame, self.child_frame )

        if berry_transform.header.frame_id == self.parent_frame: 
            rospy.loginfo("Moving to align XY" )
            self.move_pre_grasp_align_pose_xy(berry_transform)
            rospy.loginfo("Moving to approach Z" )
            self.move_pre_grasp_pose_z(berry_transform)
            _ = input ("Press y to move to ready or n to exit\n")
            if _.lower() == "y": 
                self.move_to_named_pose( pose = "start" )     
        sys.exit(0)


    def get_berry_frame(self, parent_frame, child_frame): 
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)
        transform = TransformStamped()
        not_found = True
        count = 0
        while not_found: 
            try: 
                transform = tf_buffer.lookup_transform(parent_frame, child_frame, rospy.Time())
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                count += 1
                self.rate.sleep()
                if (count > 10): 
                    break

        return transform

    def move_pre_grasp_align_pose_xy( self, transform ):

        current_pose = self.group.get_current_pose()
        current_pose.header.frame_id = transform.header.frame_id


        current_pose.pose.position.z = transform.transform.translation.z
        current_pose.pose.position.y = transform.transform.translation.y
        # current_pose.pose.position.x = transform.transform.translation.x

        self.group.clear_pose_targets()
        self.group.set_pose_target( current_pose ) 
        self.group.go(wait = True)

    def move_pre_grasp_pose_z( self, transform ):

        current_pose = self.group.get_current_pose()
        current_pose.header.frame_id = transform.header.frame_id
        current_pose.pose.position.x = transform.transform.translation.x - self.offset_z 
        self.group.clear_pose_targets()
        self.group.set_pose_target( current_pose ) 
        self.group.go(wait = True)
    
    def move_to_named_pose (self, pose = "start" ) : 

        rospy.loginfo("Moving to {} pose".format(pose) )
        self.group.clear_pose_targets() 
        self.group.set_named_target( pose ) 
        self.group.go(wait = True) 

    def move_look_at_camera(self): 

        current_pose = self.group.get_current_pose()
        current_pose.header.frame_id = transform.header.frame_id
        current_pose.pose.position.x = transform.transform.translation.x - self.offset_z 
        self.group.clear_pose_targets()
        self.group.set_pose_target( current_pose ) 
        self.group.go(wait = True)

if __name__ == '__main__':

    fastPickActionServer = FastPickActionServer()