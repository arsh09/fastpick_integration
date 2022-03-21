#!/usr/bin/env python3

'''
Muhammad Arshad 
03/03/2022
'''
from typing import List, Tuple, Dict, Set
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import Pose2D, Point, PoseStamped, TransformStamped
from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import Bool
from std_srvs.srv import Empty, EmptyResponse, EmptyRequest
import numpy as np
import tf2_ros
import time 

class FastPickPickingAndPlacingNode: 

    def __init__(self) -> None: 

        rospy.init_node("fastpick_action_server_node")
        moveit_commander.roscpp_initialize(sys.argv)

        # read params 
        arm_group_name = rospy.get_param("~arm_group", "panda_arm")
        gripper_group_name = rospy.get_param("~gripper_group", "hand")
        self.planner_id = rospy.get_param("~planner_id", "RRTConnect")
        self.planner_pipeline_id = rospy.get_param("~pipeline_id", "ompl")
        self.howmany = rospy.get_param("~how_many", 0)

        # bringup robot, arm and gripper groups and scene
        self.robot = moveit_commander.RobotCommander()
        self.group_arm = moveit_commander.MoveGroupCommander(arm_group_name)
        if gripper_group_name != "":
            self.group_hand = moveit_commander.MoveGroupCommander(gripper_group_name)
    
        self.scene = moveit_commander.PlanningSceneInterface()
        self.display_trajectory_publisher = rospy.Publisher( "/move_group/display_planned_path", moveit_msgs.msg.DisplayTrajectory, queue_size=20)
        self.robot = moveit_commander.RobotCommander()

        # set plannar (for arm) 
        self.parent_frame = self.group_arm.get_planning_frame()
        self.group_arm.set_planning_pipeline_id(self.planner_pipeline_id) 
        self.group_arm.set_planner_id(self.planner_id) 
        self.group_arm.set_max_velocity_scaling_factor(0.1)
        self.group_arm.set_max_acceleration_scaling_factor(0.1)
        self.group_arm.set_planning_time(30)
        self.group_arm.set_num_planning_attempts(10)

        # set planner (for gripper ) 
        self.group_hand.set_planning_pipeline_id("ompl") 
        self.group_hand.set_planner_id("RRTConnectkConfigDefault") 
       
        # clear octomap service 
        rospy.wait_for_service("/clear_octomap")
        self.map_service = rospy.ServiceProxy("/clear_octomap", Empty)

        # panda controller topic 
        self.panda_arm_controller = rospy.Publisher("/position_joint_trajectory_controller/command", JointTrajectory, queue_size = 1 )

        # rate used for Tf listener
        self.rate = rospy.Rate(10)

        # positon biases.        
        self.offset_x  = -0.05
        self.offset_y  = 0.025
        self.offset_z  = 0.025
        

        # this publisher stop tf for berry perception 
        self.stop_tf_pub = rospy.Publisher("/fastpick_perception/stop_perception", Bool, queue_size = 10 )

        # picking starts here.
        self.move_to_named_pose_gripper(pose="open")
        self.pick_and_place_berry_all(n = self.howmany)

    def clear_octomap(self) -> None:
        ''' clear OctoMap service client. 
        ''' 
        rospy.loginfo("Clearing octomap...")
        try: 
            self.map_service()
        except rospy.ServiceException as e:
            rospy.logerr("Clear octomap service call failed: {}".format(e) )

    def send_movements_to_controller( self, traj_in : JointTrajectory() ):
        rospy.loginfo("Using controllers directly")        
        self.group_arm.stop()
        time.sleep(5)
        for i in range(10):
            self.panda_arm_controller.publish( traj_in )
            time.sleep(2)
 

    def get_berry_frames_transforms(self, parent_frame : str, child_frames_list: List[str] ) -> Dict[str, TransformStamped]:
        ''' find transform of each berry in child_frames_list to parent_frame 

        :param: parent_frame - usually the fixed link for your robot (world, panda_link0 etc)
        :param: child_frame_lists - list of berry_frame names such as berry_{n} where n can be 1,2,3...
        '''
        transforms = {}
        for child_frame in child_frames_list:
            tf_buffer = tf2_ros.Buffer( cache_time = rospy.Duration(60) )
            tf_listener = tf2_ros.TransformListener(tf_buffer)
            not_found = True
            count = 0
            rospy.loginfo("Finding transform from {} to {}".format(parent_frame, child_frame))
            while not_found: 
                try: 
                    transform = tf_buffer.lookup_transform(parent_frame, child_frame, rospy.Time(), rospy.Duration(5))
                    transforms[child_frame] = transform
                    break
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    count += 1
                    self.rate.sleep()
                    if (count > 10): 
                        rospy.logerr("Transform is not found from {} to {}".format(parent_frame, child_frame))
                        break

        return transforms

    def move_to_pose_xyz( self, transform : TransformStamped(), x, y, z ) -> bool:
        ''' move the robot from current state to berry transform with offsets.
        '''
        self.group_arm.set_planning_pipeline_id(self.planner_pipeline_id) 
        self.group_arm.set_planner_id(self.planner_id) 

        current_pose = self.group_arm.get_current_pose()
        current_pose.header.frame_id = transform.header.frame_id        
        current_pose.pose.position.x = transform.transform.translation.x + x
        current_pose.pose.position.y = transform.transform.translation.y + y
        current_pose.pose.position.z = transform.transform.translation.z + z
        self.group_arm.stop()
        self.group_arm.set_start_state_to_current_state()
        self.group_arm.clear_pose_targets()
        self.group_arm.set_pose_target( current_pose ) 
        retval = self.group_arm.go(wait = True)
 
        return retval
    
    def move_to_named_pose_arm (self, pose : str = "ready" ) -> bool: 
        ''' moves the robot arm to a previously saved named pose.
        :param pose - saved named pose in moveit config SRDF file
        '''
        # set planner (for arm) 
        self.group_arm.set_planning_pipeline_id("ompl") 
        self.group_arm.set_planner_id("RRTConnectkConfigDefault") 

        rospy.loginfo("Moving arm to {} pose".format(pose) )
        self.group_arm.clear_pose_targets() 
        self.group_arm.set_named_target( pose ) 
        retval = self.group_arm.go(wait = True) 

    def move_to_named_pose_gripper (self, pose = "open" ) -> bool: 
        ''' moves the robot gripper to a previously saved named pose. 
        :param pose- saved named pose in moveit config SRDF file.
        '''
        rospy.loginfo("Moving gripper to {} pose".format(pose) )
        self.group_hand.clear_pose_targets() 
        self.group_hand.set_named_target( pose ) 
        retval = self.group_hand.go(wait = True) 
 
    def pick_and_place_berry_all(self, n : int = 0) -> None: 
        ''' start the pick and place pipeline and attempts 
            pick all the detected berries.
        :param n - number of berries to pick from 1,2,3,4... 
        '''
        if n > 0: 
            self.clear_octomap()
            berry_list = {}        
            berry_name_list = []
            for i in range(1, n+1): 
                child_frame = "berry_{}".format(i)
                berry_name_list.append( child_frame )
            
            berry_list = self.get_berry_frames_transforms(self.parent_frame, berry_name_list)

            self.move_to_named_pose_gripper(pose = "open")
            rospy.loginfo("We are going to pick {} berries in this cycle.".format(n))

            # Stop TF publisher for perception.
            # Reason in fastpick_perception/nodes/fastpick_filter_mask_node.py
            msg = Bool() 
            msg.data = True
            self.stop_tf_pub.publish( msg ) 
            for berry in berry_list.keys(): 
                try: 
                    rospy.loginfo("Picking {}".format(berry))
                    self.pick_and_place_berry( berry_list[berry], berry )
                    rospy.loginfo("---------------------------------------------------------------")
                    time.sleep(1)
                except KeyboardInterrupt: 
                    rospy.logerr("Error in picking seqeunce")
                    break
            
            # Start TF publisher for perception.
            msg.data = False
            self.stop_tf_pub.publish( msg )             

    def pick_and_pkace_cartesian_path( self, x, y, z, disable_collision : bool = True):
        ''' calculate cartesian path beteen start, intermediate and end point. 

        :param berry_pose is end pose 
        '''
        # set planner (for arm) 
        self.group_arm.set_planning_pipeline_id("ompl") 
        self.group_arm.set_planner_id("RRTConnectkConfigDefault") 

        start_pose = self.group_arm.get_current_pose()
        waypoints = []
        waypoints.append(copy.deepcopy(start_pose.pose))

        start_pose.pose.position.x += x
        start_pose.pose.position.y += y
        start_pose.pose.position.z += z
        waypoints.append(copy.deepcopy(start_pose.pose))
        rospy.loginfo("Computing cartesian path")
        (plan, fraction) = self.group_arm.compute_cartesian_path(
            waypoints, 0.01, 0.0 , # waypoints to follow  # eef_step
            avoid_collisions = False,
        )  # jump_threshold

        rospy.loginfo("Computed cartesian path with fraction of {}".format(fraction))
        
        rospy.loginfo("Retiming the computed trajectory")
        plan_out = self.group_arm.retime_trajectory( self.robot.get_current_state(), plan,
            velocity_scaling_factor=0.1,
            acceleration_scaling_factor=0.2,
            algorithm="time_optimal_trajectory_generation",
        )

        return plan_out, fraction

    def move_to_approach_pose(self, x, y, z):

        self.group_arm.stop()
        self.group_arm.clear_pose_targets()
        plan, fraction = self.pick_and_pkace_cartesian_path( x, y, z, disable_collision = True )             
        retval = self.group_arm.execute( plan, wait = True )
        if not retval : 
            rospy.logwarn("Unable to approach fully. Plan fraction is {}".format(retval, fraction))
        else: 
            rospy.loginfo("Approached berry fully. Plan fraction is {}".format(retval, fraction))

    def move_to_retract_pose(self, x, y, z): 
        self.group_arm.stop()
        self.group_arm.clear_pose_targets()
        plan, fraction = self.pick_and_pkace_cartesian_path( x, y, z, disable_collision = True )             
        retval = self.group_arm.execute( plan, wait = True )
        if not retval : 
            rospy.logwarn("Unable to retract fully. Plan fraction is {}".format(retval, fraction))
        else: 
            rospy.loginfo("Retracted back from berry fully. Plan fraction is {}".format(retval, fraction))

    def pick_and_place_berry(self, berry_pose : TransformStamped(), berry_frame : str ) -> None:
        ''' A single manipulation cycle from robot home pose -> pre-grasp-pose -> close-gripper etc. 

        :param: berry_pose is berry pose w.r.t arm-group planning frame 
        :param: berry_frame is berry TF frame name. 
        ''' 
        if berry_pose.header.frame_id == self.parent_frame: 
            # self.clear_octomap()
            rospy.loginfo("Moving to {} ".format(berry_frame) )
            is_picked = self.move_to_pose_xyz(berry_pose, self.offset_x, self.offset_y, self.offset_z )
            if is_picked: 
                self.move_to_approach_pose( 0.075, 0.0, 0.0 )
                self.move_to_named_pose_gripper(pose = "close")
                self.move_to_named_pose_gripper(pose = "open")
                self.move_to_retract_pose( -0.075, 0.0, 0.0 )

                rospy.loginfo("Pick {} successfully. Moving to picking pose".format(berry_frame))
                self.move_to_named_pose_arm( pose = "pick_start" )     
                time.sleep(1)
                self.move_to_named_pose_arm( pose = "place_start" )
                time.sleep(1)
                self.move_to_named_pose_arm( pose = "pick_start" )     
                time.sleep(1)
                self.clear_octomap()
            else: 
                self.move_to_named_pose_arm( pose = "pick_start" )     
                time.sleep(1)
                rospy.loginfo("Pick {} unsuccessful. We will try to pick it in next cycle. Moving to picking pose".format(berry_frame))

        else: 
            rospy.logerr("Berry frame_id does not match fixed frame of the arm group. Please check the transformation.")


if __name__ == '__main__':

    fastPickPickingAndPlacingNode = FastPickPickingAndPlacingNode()