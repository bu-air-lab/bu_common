#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import rospy

from location import *
from controller import *

from os import path


if __name__ == "__main__":
    try: 
        rospy.init_node("showroom_demo")
        ctrl = ControllerClass()
        sound_handler = SoundClient()
        rospy.sleep(rospy.Duration.from_sec(1.0))
        # mixer.init()
        sound_handler.say("Program starts.") 
        rospy.sleep(rospy.Duration.from_sec(2.0))
        sound_handler.say("I'm moving to the initial location.")
        ctrl.goto_location(locations["corridor"]) 
        rospy.sleep(10.0)
        sound_handler.say("\
            Hello. \
            Welcome to the house. \
            I am going to offer A tour for you. \
            This house has a number of desirable features."
        )
        rospy.sleep(10.0)
        # ######
        # # From corridor to Kitchen
        # ######
        # sound_handler.say("Now we are at a corridor of the house. \
        #     Please let me show you the kitchen first."
        # )
        # rospy.sleep(8.0)
        # sound_handler.say("Please follow me.")
        # ctrl.goto_location(locations["kitchen"])
        # sound_handler.say("We arrived the kitchen. \
        #     You can see the kitchen is spacious. \
        #     You can really enjoy cooking with this kitchen."
        # )
        # rospy.sleep(10.0)
        # sound_handler.say("Do you want to know more about the kitchen?")
        # rospy.sleep(5.0)

        # ######
        # # From Kitchen to Room
        # ######
        # sound_handler.say("Okay. \
        #     Let me show you the private room with good scenary.")
        # rospy.sleep(8.0)
        # sound_handler.say("Please follow me.")
        # ctrl.goto_location(locations["room"])  
        # sound_handler.say("We arrived the private room. \
        #     This room is spacious and has a good scenary of a quiet residential area. \
        #     It's nice to spend some time reading books in this quiet place."
        # )

        ######
        # From corridor to Private room
        ######
        sound_handler.say("Now we are at a corridor of the house. \
            Please let me show you the private room with good scenary first."
        )
        rospy.sleep(8.0)
        sound_handler.say("Please follow me.")
        ctrl.goto_location(locations["room"])
        sound_handler.say("We arrived the private room. \
            This room is spacious and has a good scenary of a quiet residential area. \
            It's nice to spend some time reading books in this quiet place."
        )
        rospy.sleep(18.0)
        sound_handler.say("Do you want to know more about the private room?")
        rospy.sleep(7.0)

        ######
        # From Kitchen to Room
        ######
        sound_handler.say("Okay. \
            Let me show you the kitchen next.")
        rospy.sleep(8.0)
        sound_handler.say("Please follow me.")
        ctrl.goto_location(locations["room"])  
        sound_handler.say("We arrived the kitchen. \
            You can see the kitchen is spacious. \
            You can really enjoy cooking with this kitchen."
        )
        
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("Finished with interrupt")