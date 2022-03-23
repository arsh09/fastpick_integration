#! /usr/bin/env python

'''
An action server wrapper around fastpick pick and place pipeline 

Muhammad Arshad
23/03/2022
'''
from typing import List, Tuple, Dict, Set
import rospy
import sys
import rospy
import actionlib 
from actionlib_msgs.msg import GoalID
from fastpick_msgs.msg import StrawberryPickAction, StrawberryPickGoal, StrawberryPickFeedback, StrawberryPickResult


class FastPickAndPlaceActionClient():

    def __init__(self): 

        try:
            rospy.init_node('action_client')
            result = self.call_server()
            rospy.loginfo("Results are: Picked Count: {} Unpicked Count: {}".format(result.picked, result.not_picked))
        except rospy.ROSInterruptException as e:
            rospy.logerr("Something went wrong. {}".format(e))

    def call_server(self):

        client = actionlib.SimpleActionClient('fastpick_pick_and_place', StrawberryPickAction)
        client.wait_for_server()

        goal = StrawberryPickGoal()
        goal.num_berries_to_pick = rospy.get_param("~how_many", 1)

        client.send_goal(goal, feedback_cb=self.feedback_cb)
        client.wait_for_result()

        result = client.get_result()

        return result

    def feedback_cb(self, feedback):
        rospy.loginfo("Status: {}. Picked: {}. Unpicked: {}".format(feedback.status, feedback.picked_till_now, feedback.pick_remains))


if __name__ == "__main__": 

    client = FastPickAndPlaceActionClient()
