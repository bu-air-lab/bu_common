#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from builtins import (str, open, object)

import rospy
import subprocess
# from pygame import mixer
from threading import Timer

import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import *
from geometry_msgs.msg import Point, PoseWithCovarianceStamped, PoseStamped

import std_srvs.srv 
from nav_msgs.srv import GetPlan

from sound_play.libsoundplay import SoundClient

from location import locations
from collections import defaultdict


class ControllerClass(object):
    def __init__(self):
        self.to_x, self.to_y = None, None
        self.pose_x, self.pose_y = None, None

        self.action_client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        while not self.action_client.wait_for_server(rospy.Duration.from_sec(5.0)):
            rospy.loginfo("Waiting for the move_base action server to come up")
        
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.callback_pose)

        self.clear_cost = rospy.ServiceProxy('/move_base/clear_costmaps', std_srvs.srv.Empty())
        
        rospy.sleep(3.0)
        
    def clear_obstacle_map(self):
        self.clear_cost()
        rospy.sleep(3.0)

    def goto_location(self, location):
        is_goal_reached = self.move_to_goal(location)
        return is_goal_reached

    def move_to_goal(self, location):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        self.to_x, self.to_y = location["x"], location["y"]
        ox, oy, oz, ow = location["ox"], location["oy"], location["oz"], location["ow"]
        goal.target_pose.pose.position = Point(self.to_x, self.to_y, 0.0)
        goal.target_pose.pose.orientation.x = ox
        goal.target_pose.pose.orientation.y = oy
        goal.target_pose.pose.orientation.z = oz
        goal.target_pose.pose.orientation.w = ow
        rospy.loginfo("Sending location ({}, {})".format(self.to_x, self.to_y))
        self.action_client.send_goal(goal, self.done_callback, self.active_callback, self.feedback_callback)

        self.action_client.wait_for_result(rospy.Duration(360))
        if self.action_client.get_state() == GoalStatus.SUCCEEDED:
            return True
        else:
            self.action_client.cancel_all_goals()
            return False

    def active_callback(self):
        pass

    def done_callback(self, status, result):
        pass

    def feedback_callback(self, feedback):
        pass

    def callback_pose(self, data):
        self.pose_x = data.pose.pose.position.x
        self.pose_y = data.pose.pose.position.y
        with open('robot_location.txt', 'a') as f:
            f.write(str(self.pose_x) + "," + str(self.pose_y))
            f.close()