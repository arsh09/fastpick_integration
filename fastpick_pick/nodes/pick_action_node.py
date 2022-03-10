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
from std_srvs.srv import Empty, EmptyResponse, EmptyRequest
import numpy as np
import tf2_ros
import time 

class FastPickActionServer: 

    def __init__(self): 

            
        rospy.init_node("fastpick_action_server_node")
    
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        group_name = "panda_arm"
        self.group_arm = moveit_commander.MoveGroupCommander(group_name)
        group_name = "hand"
        self.group_hand = moveit_commander.MoveGroupCommander(group_name)
        self.scene = moveit_commander.PlanningSceneInterface()
        self.display_trajectory_publisher = rospy.Publisher( "/move_group/display_planned_path", moveit_msgs.msg.DisplayTrajectory, queue_size=20)
        self.planned_path = None

        self.group_arm.set_max_velocity_scaling_factor(0.2)
        self.group_arm.set_max_acceleration_scaling_factor(0.75)
        self.group_arm.allow_looking(True)
        # self.group_arm.set_planning_time(30)
        # self.group_arm.set_num_planning_attempts(10)

        self.parent_frame = self.group_arm.get_planning_frame()

        # clear octomap service 
        rospy.wait_for_service("/clear_octomap")
        self.map_service = rospy.ServiceProxy("/clear_octomap", Empty)

        self.rate = rospy.Rate(10)
 
        self.child_frame = rospy.get_param("~berry_frame", "berry_1")
        self.howmany = rospy.get_param("~how_many", 0)
        
        self.offset_z  = 0.15
        self.pick_and_place_berry_all(n = self.howmany)
 
    def get_berry_frame(self, parent_frame, child_frame): 
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)
        transform = TransformStamped()
        not_found = True
        count = 0
        while not_found: 
            try: 
                transform = tf_buffer.lookup_transform(parent_frame, child_frame, rospy.Time())
                transform.transform.translation.x -= self.offset_z
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                count += 1
                self.rate.sleep()
                if (count > 10): 
                    break

        return transform

    def get_berry_frames_transforms(self, parent_frame, child_frames_list): 
        
        transforms = {}
        for child_frame in child_frames_list:
            tf_buffer = tf2_ros.Buffer( cache_time = rospy.Duration(60) )
            tf_listener = tf2_ros.TransformListener(tf_buffer)
            not_found = True
            count = 0
            rospy.loginfo("Finding transform from {} to {}".format(parent_frame, child_frame))
            while not_found: 
                try: 
                    # transform = tf_buffer.lookup_transform(parent_frame, child_frame, rospy.Duration(1))
                    transform = tf_buffer.lookup_transform(parent_frame, child_frame, rospy.Time(), rospy.Duration(5))
                    transform.transform.translation.x -= self.offset_z
                    transforms[child_frame] = transform
                    break
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    count += 1
                    self.rate.sleep()
                    if (count > 10): 
                        rospy.logerr("Transform is not found from {} to {}".format(parent_frame, child_frame))
                        break

        return transforms

    def move_pre_grasp_align_pose_xy( self, transform ):

        current_pose = self.group_arm.get_current_pose()
        current_pose.header.frame_id = transform.header.frame_id        

        current_pose.pose.position.x = transform.transform.translation.x 
        current_pose.pose.position.y = transform.transform.translation.y
        current_pose.pose.position.z = transform.transform.translation.z

        self.group_arm.clear_pose_targets()
        self.group_arm.set_pose_target( current_pose ) 
        self.group_arm.go(wait = True)

    def move_pre_grasp_pose_z( self, transform ):

        current_pose = self.group_arm.get_current_pose()
        current_pose.header.frame_id = transform.header.frame_id
        current_pose.pose.position.x = transform.transform.translation.x - self.offset_z 
        self.group_arm.clear_pose_targets()
        self.group_arm.set_pose_target( current_pose ) 
        self.group_arm.go(wait = True)
    
    def move_to_named_pose_arm (self, pose = "ready" ) : 

        rospy.loginfo("Moving arm to {} pose".format(pose) )
        self.group_arm.clear_pose_targets() 
        self.group_arm.set_named_target( pose ) 
        self.group_arm.go(wait = True) 

    def move_to_named_pose_gripper (self, pose = "open" ) : 
        rospy.loginfo("Moving gripper to {} pose".format(pose) )
        self.group_hand.clear_pose_targets() 
        self.group_hand.set_named_target( pose ) 
        self.group_hand.go(wait = True) 

    def clear_octomap(self): 
        rospy.loginfo("Clearing octomap...")
        try: 
            self.map_service()
        except rospy.ServiceException as e:
            rospy.logerr("Clear octomap service call failed: {}".format(e) )

    def pick_and_place_berry_all(self, n = 0): 

        if n > 0: 
            berry_list = {}        
            berry_name_list = []
            for i in range(1, n+1): 
                child_frame = "berry_{}".format(i)
                berry_name_list.append( child_frame )
                # rospy.loginfo("Finding transfrom from {} to {}".format(self.parent_frame, child_frame))
                # berry_list["berry_{}".format(i)] = berry_transform = self.get_berry_frame( self.parent_frame, child_frame )
            
            berry_list = self.get_berry_frames_transforms(self.parent_frame, berry_name_list)

            self.move_to_named_pose_gripper(pose = "open")
            rospy.loginfo("We are going to pick {} berries in this cycle.".format(n))
            for berry in berry_list.keys(): 
                try: 
                    rospy.loginfo("Picking {}".format(berry))
                    self.pick_and_place_berry( berry_list[berry], berry )
                    rospy.loginfo("------------------------------------")
                    time.sleep(1)
                except KeyboardInterrupt: 
                    rospy.logerr("Error in picking seqeunce")
                    break

    def pick_and_place_berry(self, berry_pose, berry_frame ): 
 
        if berry_pose.header.frame_id == self.parent_frame: 
            # self.clear_octomap()
            rospy.loginfo("Moving to {} ".format(berry_frame) )
            self.move_pre_grasp_align_pose_xy(berry_pose)
            self.move_to_named_pose_gripper(pose = "close")
            self.move_to_named_pose_gripper(pose = "open")
            rospy.loginfo("Pick {} successfully. Moving to picking pose".format(berry_frame))
            self.move_to_named_pose_arm( pose = "pick_start" )     
            time.sleep(5)
        else: 
            pass

 
if __name__ == '__main__':

    fastPickActionServer = FastPickActionServer()