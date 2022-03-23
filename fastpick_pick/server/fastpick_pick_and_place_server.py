#! /usr/bin/env python

'''
An action server wrapper around fastpick pick and place pipeline 

Muhammad Arshad
23/03/2022
'''

from typing import List, Tuple, Dict, Set
import copy
import sys
import rospy
import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import  TransformStamped
from std_msgs.msg import Bool
from std_srvs.srv import Empty
import numpy as np
import tf2_ros
import time 
import actionlib 
from actionlib_msgs.msg import GoalID
from fastpick_msgs.msg import StrawberryPickAction, StrawberryPickGoal, StrawberryPickFeedback, StrawberryPickResult

class FastPickMoveIt(): 

    def __init__(self, config : dict ) -> None: 
        ''' starts moveit command and other stuff.

        :param (config, dict) contains params
        '''

        self.config = config 

        # moveit commander initialize
        moveit_commander.roscpp_initialize(sys.argv)

        self.log = True 
        
        # bringup robot, arm and gripper groups and scene
        self.robot = moveit_commander.RobotCommander()
        all_group_names = self.robot.get_group_names()

        if self.config["arm_group_name"]  in all_group_names: 
            self.group_arm = moveit_commander.MoveGroupCommander(self.config["arm_group_name"])
            self.set_planner_and_pipeline( self.group_arm, self.config["pipeline_id"] , self.config["pipeline_id"] )
        
        if self.config["gripper_group_name"] in all_group_names: 
            self.group_gripper = moveit_commander.MoveGroupCommander(self.config["gripper_group_name"])
            self.set_planner_and_pipeline( self.group_gripper, "ompl" , "RRTConnectkConfigDefault" )

        # scene and display traj publisher
        self.scene = moveit_commander.PlanningSceneInterface()
        self.display_trajectory_publisher = rospy.Publisher( "/move_group/display_planned_path", moveit_msgs.msg.DisplayTrajectory, queue_size=20)

        # cancel moveit trajectory goal
        self._cancel_movement = rospy.Publisher("/execute_trajectory/cancel", GoalID, queue_size = 10)
        
        # clear octomap service 
        rospy.wait_for_service("/clear_octomap")
        self.map_service = rospy.ServiceProxy("/clear_octomap", Empty)

        # this publisher stop tf for berry perception 
        self.perception_stop_topic = rospy.Publisher("/fastpick_perception/stop_perception", Bool, queue_size = 10 )

    def set_planner_and_pipeline(self, group, pipeline, planner ) : 
        group.set_planning_pipeline_id(pipeline)
        group.set_planner_id(planner)
        group.set_max_acceleration_scaling_factor(1.0)
        group.set_max_velocity_scaling_factor(1.0)

    def abort_movement(self) -> None: 
        ''' abort any movement if the pick berry goal 
            receives any cancel request 
        '''
        goalId = GoalID()
        goalId.stamp = rospy.Time().now()
        self._cancel_movement.publish(GoalID)

    def move_to_named_pose_arm (self, pose : str = "ready" ) -> bool: 
        ''' moves the robot arm to a previously saved named pose.
        :param pose - saved named pose in moveit config SRDF file
        '''
        # set planner (for arm) 
        self.set_planner_and_pipeline( self.group_arm, "ompl" , "RRTConnectkConfigDefault" )
        if self.log: rospy.loginfo("Moving arm to {} pose".format(pose) )
        self.group_arm.clear_pose_targets() 
        self.group_arm.set_named_target( pose ) 
        retval = self.group_arm.go(wait = True) 

    def move_to_named_pose_gripper (self, pose = "open" ) -> bool: 
        ''' moves the robot gripper to a previously saved named pose. 
        :param pose- saved named pose in moveit config SRDF file.
        '''
        if self.log: rospy.loginfo("Moving gripper to {} pose".format(pose) )
        self.group_gripper.clear_pose_targets() 
        self.group_gripper.set_named_target( pose ) 
        retval = self.group_gripper.go(wait = True) 
 
    def move_to_pose_xyz( self, transform : TransformStamped(), x, y, z ) -> bool:
        ''' move the robot from current state to berry transform with offsets.
        '''
        self.set_planner_and_pipeline( self.group_arm, self.config["pipeline_id"] , self.config["planner_id"] )
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

    def pick_and_place_cartesian_path( self, x : float, y : float, z : float, disable_collision : bool = True):
        ''' calculate cartesian path beteen start, intermediate and end point. 
        '''
        self.set_planner_and_pipeline( self.group_arm, "ompl" , "RRTConnectkConfigDefault" )
        start_pose = self.group_arm.get_current_pose()
        waypoints = []
        waypoints.append(copy.deepcopy(start_pose.pose))
        start_pose.pose.position.x += x
        start_pose.pose.position.y += y
        start_pose.pose.position.z += z
        waypoints.append(copy.deepcopy(start_pose.pose))
        (plan, fraction) = self.group_arm.compute_cartesian_path(
            waypoints, 0.01, 0.0 ,  
            avoid_collisions = disable_collision,
        )   

        if self.log: rospy.loginfo("Computed cartesian path with fraction of {}".format(fraction))
        if self.log: rospy.loginfo("Retiming the computed trajectory")
        plan_out = self.group_arm.retime_trajectory( self.robot.get_current_state(), plan,
            velocity_scaling_factor=0.1,
            acceleration_scaling_factor=0.2,
            algorithm="time_optimal_trajectory_generation",
        )

        return plan_out, fraction

    def move_to_approach_pose(self, x : float, y : float, z : float) -> None:
        ''' move the arm to approch the berry in straight x-direction
        '''
        self.group_arm.stop()
        self.group_arm.clear_pose_targets()
        plan, fraction = self.pick_and_place_cartesian_path( x, y, z, disable_collision = True )             
        retval = self.group_arm.execute( plan, wait = True )
        if not retval : 
            if self.log: rospy.logwarn("Unable to approach fully. Plan fraction is {}".format(retval, fraction))
        else: 
            if self.log: rospy.loginfo("Approached berry fully. Plan fraction is {}".format(retval, fraction))

    def move_to_retract_pose(self, x : float, y : float, z : float): 
        ''' move the arm to retract from the berry in straight x-direction
        '''
        self.group_arm.stop()
        self.group_arm.clear_pose_targets()
        plan, fraction = self.pick_and_place_cartesian_path( x, y, z, disable_collision = True )             
        retval = self.group_arm.execute( plan, wait = True )
        if not retval : 
            if self.log: rospy.logwarn("Unable to retract fully. Plan fraction is {}".format(retval, fraction))
        else: 
            if self.log: rospy.loginfo("Retracted back from berry fully. Plan fraction is {}".format(retval, fraction))

    def clear_octomap(self):
        ''' clear OctoMap service client. 
        ''' 
        try: 
            self.map_service()
            return True
        except rospy.ServiceException as e:
            return False


    def stop_perception(self): 
        msg = Bool() 
        msg.data = True
        self.perception_stop_topic.publish( msg ) 

    def start_perception(self): 
        msg = Bool() 
        msg.data = False
        self.perception_stop_topic.publish( msg ) 

class FastPickAndPlaceAction(object): 

    def __init__(self, name): 

        config = {
            "arm_group_name" : "panda_arm",
            "gripper_group_name" : "hand",
            "pipeline_id" : "pilz_industrial_motion_planner",
            "planner_id" : "PTP",
        }

        # positon biases.        
        self.offset_x  = 0.035
        self.offset_y  = -0.025
        self.offset_z  = -0.015
        
        self.fastpick_arm = FastPickMoveIt(config)
        self._action_name = name
        self._server = actionlib.SimpleActionServer(self._action_name, StrawberryPickAction, execute_cb=self.execute_cb, auto_start = False)
        self.rate = rospy.Rate(100)
        self._server.start()
    
    def execute_cb(self, goal): 
        
        self._feedback = StrawberryPickFeedback()
        self._result = StrawberryPickResult() 
        self._goal = StrawberryPickGoal()
        self.num_picked_success = 0
        self.num_picked_fail = 0

        self._goal = goal 
        self.success = True
        self.is_robot_moving = False

        berry_list = {}
        for i in range(1, self._goal.num_berries_to_pick + 1):
            if i not in goal.berry_to_ignore:
                child_frame = "berry_{}".format(i)
                berry_list[child_frame] = None
        
        berries_with_tf = self.get_berry_frames_transforms( "panda_link0", berry_list)
        status = "Started picking sequence now" 
        self.send_feedback ( status = status )
        self.is_robot_moving = True
        self.fastpick_arm.stop_perception()
        self.pick_and_place_berry_all(berries_with_tf)
        self.fastpick_arm.start_perception()

        if self.success:
            self._result.total = self._goal.num_berries_to_pick
            self._result.picked = self.num_picked_success
            self._result.not_picked = self.num_picked_fail
            rospy.loginfo("{} goal successded".format(self._action_name))
            self._server.set_succeeded(self._result)
        else: 
            rospy.loginfo("{} goal cancelled".format(self._action_name))

    def send_feedback(self, status = "", picked_till_now = 0, pick_remains = 0 ):
        self._feedback.status = status
        self._feedback.picked_till_now = picked_till_now
        self._feedback.pick_remains = pick_remains
        self._server.publish_feedback(self._feedback)

    def is_goal_cancel(self): 
        if self._server.is_preempt_requested():
            rospy.loginfo("Preempting the goal for action {}".format(self._action_name))
            self._server.set_preempted()
            self.success = False
            if self.is_robot_moving : 
                self.fastpick_arm.abort_movement()
                self.fastpick_arm.start_perception()
            return True
        else: 
            return False

    def get_berry_frames_transforms(self, parent_frame : str, child_frames_list: Dict[str, TransformStamped] ) -> Dict[str, TransformStamped]:
        ''' find transform of each berry in child_frames_list to parent_frame 

        :param: parent_frame - usually the fixed link for your robot (world, panda_link0 etc)
        :param: child_frame_lists - dict of berry_frame names such as berry_{n} where n can be 1,2,3... as keys
        '''
        for child_frame in child_frames_list.keys():
            tf_buffer = tf2_ros.Buffer( cache_time = rospy.Duration(60) )
            tf_listener = tf2_ros.TransformListener(tf_buffer)
            not_found = True
            count = 0
            rospy.loginfo("Finding transform from {} to {}".format(parent_frame, child_frame))
            while not_found:
                if self.is_goal_cancel():
                    break 
                try: 
                    transform = tf_buffer.lookup_transform(parent_frame, child_frame, rospy.Time(), rospy.Duration(5))
                    child_frames_list[child_frame] = transform
                    status = "Found transform from {} to {}".format(parent_frame, child_frame)
                    self.send_feedback(status = status)
                    break
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    count += 1
                    self.rate.sleep()
                    if (count > 10): 
                        status = "Unable to find transform from {} to {}. Adding it to ignore berry.".format(parent_frame, child_frame)
                        self.send_feedback(status = status)
                        break

        return child_frames_list


    def pick_and_place_berry_all(self, child_frames_list: Dict[str, TransformStamped]) -> None: 
        ''' start the pick and place pipeline and attempts 
            pick all the detected berries.
        :param: child_frame_lists - dict of berry_frame names such as berry_{n} where n can be 1,2,3... as keys
        '''
        for berry in child_frames_list.keys(): 
            try: 
                if not isinstance( child_frames_list[berry], type(None) ):
                    self.pick_and_place_berry( child_frames_list[berry], berry )
            except KeyboardInterrupt: 
                break
        

    def pick_and_place_berry( self, berry_pose : TransformStamped(), berry_frame : str ) -> None: 

        ''' A single manipulation cycle from robot home pose -> pre-grasp-pose -> close-gripper etc. 

        :param: berry_pose is berry pose w.r.t arm-group planning frame 
        :param: berry_frame is berry TF frame name. 
        ''' 

        status = "Clearing octomap..."
        self.send_feedback ( status = status , picked_till_now = self.num_picked_success , pick_remains = self._goal.num_berries_to_pick - (self.num_picked_success + self.num_picked_fail) )              
        self.fastpick_arm.clear_octomap()

        if not self.is_goal_cancel(): 
            status = "Moving to pick {}".format(berry_frame)
            self.send_feedback ( status = status , picked_till_now = self.num_picked_success , pick_remains = self._goal.num_berries_to_pick - (self.num_picked_success + self.num_picked_fail) ) 
            is_picked = self.fastpick_arm.move_to_pose_xyz(berry_pose, self.offset_x, self.offset_y, self.offset_z )

        if is_picked: 
            
            if not self.is_goal_cancel(): 
                status = "Approaching {}".format(berry_frame) 
                self.send_feedback ( status = status , picked_till_now = self.num_picked_success , pick_remains = self._goal.num_berries_to_pick - (self.num_picked_success + self.num_picked_fail) ) 
                self.fastpick_arm.move_to_approach_pose( 0.075, 0.0, 0.0 )
            
            if not self.is_goal_cancel(): 
                status = "Closing gripper to grasp {}".format(berry_frame) 
                self.send_feedback ( status = status , picked_till_now = self.num_picked_success , pick_remains = self._goal.num_berries_to_pick - (self.num_picked_success + self.num_picked_fail) ) 
                self.fastpick_arm.move_to_named_pose_gripper(pose = "close")
            
            if not self.is_goal_cancel(): 
                status = "Opening gripper to ungrasp {}".format(berry_frame) 
                self.send_feedback ( status = status , picked_till_now = self.num_picked_success , pick_remains = self._goal.num_berries_to_pick - (self.num_picked_success + self.num_picked_fail) ) 
                self.fastpick_arm.move_to_named_pose_gripper(pose = "open")
            
            if not self.is_goal_cancel(): 
                status = "Retracting from {}".format(berry_frame) 
                self.send_feedback ( status = status , picked_till_now = self.num_picked_success , pick_remains = self._goal.num_berries_to_pick - (self.num_picked_success + self.num_picked_fail) ) 
                self.fastpick_arm.move_to_retract_pose( -0.075, 0.0, 0.0 )
            
            if not self.is_goal_cancel(): 
                status = "Moving to pick_start pose"
                self.send_feedback ( status = status , picked_till_now = self.num_picked_success , pick_remains = self._goal.num_berries_to_pick - (self.num_picked_success + self.num_picked_fail) ) 
                self.fastpick_arm.move_to_named_pose_arm( pose = "pick_start" )     
            
            if not self.is_goal_cancel(): 
                status = "Moving to place_start pose"
                self.send_feedback ( status = status , picked_till_now = self.num_picked_success , pick_remains = self._goal.num_berries_to_pick - (self.num_picked_success + self.num_picked_fail) ) 
                self.fastpick_arm.move_to_named_pose_arm( pose = "place_start" )
            
            if not self.is_goal_cancel(): 
                status = "Moving to pick_start pose"
                self.send_feedback ( status = status , picked_till_now = self.num_picked_success , pick_remains = self._goal.num_berries_to_pick - (self.num_picked_success + self.num_picked_fail) ) 
                self.fastpick_arm.move_to_named_pose_arm( pose = "pick_start" )     
 
            self.num_picked_success += 1
        else: 
            status = "Unsuccessful pick. Moving to pick_start pose"
            self.send_feedback ( status = status , picked_till_now = self.num_picked_success , pick_remains = self._goal.num_berries_to_pick - (self.num_picked_success + self.num_picked_fail) ) 
            if not self.is_goal_cancel():self.fastpick_arm.move_to_named_pose_arm( pose = "pick_start" )     
            self.num_picked_fail += 1

if __name__ == '__main__':
    rospy.init_node('fastpick_pick_and_place')
    server = FastPickAndPlaceAction(rospy.get_name())
    rospy.spin()
