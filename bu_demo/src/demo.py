#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import rospy

from location import *
from controller import *

from os import path


if __name__ == "__main__":
    try: 
        rospy.init_node("bu_demo")
        ctrl = ControllerClass()
        sound_handler = SoundClient()
        rospy.sleep(rospy.Duration.from_sec(1.0))
        # mixer.init()
        sound_handler.say("Program starts.") 
        rospy.sleep(rospy.Duration.from_sec(2.0))
        sound_handler.say("I'm moving to the initial location.")
        ctrl.goto_location(locations["point2"]) 
        rospy.sleep(10.0)
        sound_handler.say("\
            Hello. \
            Welcome to the AIR Lab. \
            I am going to show my demonstration for you."
        )
        rospy.sleep(9.0)

        # ######
        # # From corridor to Private room
        # ######
        # sound_handler.say("In this demo, I am going to move around this lab.")
        # rospy.sleep(8.0)
        # sound_handler.say("Please keep your eyes on me.")
        # rospy.sleep(8.0)
        ctrl.goto_location(locations["point1"])
        sound_handler.say("Now you could see I was moving to here autonomously."
        )
        rospy.sleep(8.0)

        ######
        # From Kitchen to Room
        ######
        sound_handler.say("I also can avoid obstacles by using the laser sensor on my base.")
        rospy.sleep(8.0)
        sound_handler.say("I will show you this now.")
        ctrl.goto_location(locations["point2"])  
        sound_handler.say("Did I succeed in avoiding objects and you all? If not, the engineer must update the parameter of the algorithm.")


        rospy.sleep(15.0) 
        sound_handler.say("Thank you for visiting us.  I am going to move around here for a while.")
        rospy.sleep(8.0) 
        sound_handler.say("If you have any questions, please feel free to ask to students in the lab.")

        while(True):
            ctrl.goto_location(locations["point3"])
            ctrl.goto_location(locations["point2"])
            ctrl.goto_location(locations["point1"])
            ctrl.goto_location(locations["point2"])  
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("Finished with interrupt")