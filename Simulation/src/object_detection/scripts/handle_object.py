#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import numpy as np
import time
import math


class Nodo(object):
    def stop(self):
        rospy.loginfo("stop sign detected")
        self.velocities.linear.x = 0
        self.velocities.angular.z = 0
        time.sleep(5)
        self.object_handled = True
    
    def turn_left(self):
        rospy.loginfo("turn left detected")
        self.velocities.linear.x = 1
        self.velocities.angular.z = 0
        time.sleep(0.5)
        self.velocities.linear.x = 0
        self.velocities.angular.z = math.pi/4
        time.sleep(2)
        self.velocities.linear.x = 0
        self.velocities.angular.z = 0
        time.sleep(0.1)
        self.object_handled = True

    def go_straight(self):
        rospy.loginfo("go straight detected")
        self.velocities.linear.x = 0.5
        self.velocities.angular.z = 0
        time.sleep(1)
        self.velocities.linear.x = 0
        self.velocities.angular.z = 0
        time.sleep(0.1)
        self.object_handled = True

    def round_about(self):
        rospy.loginfo("round about detected")
        self.object_handled = True

    def construction_ahead(self):
        rospy.loginfo("construction ahead detected")
        self.velocities.linear.x = 0
        self.velocities.angular.z = -math.pi/4
        time.sleep(4)
        self.velocities.linear.x = 0
        self.velocities.angular.z = 0
        time.sleep(0.1)
        self.object_handled = True

    def turn_right(self):
        rospy.loginfo("turn right detected")
        self.velocities.linear.x = 1
        self.velocities.angular.z = 0
        time.sleep(0.5)
        self.velocities.linear.x = 0
        self.velocities.angular.z = -math.pi/4
        time.sleep(2)
        self.velocities.linear.x = 0
        self.velocities.angular.z = 0
        time.sleep(0.1)
        self.object_handled = True

    def give_way(self):
        rospy.loginfo("give way detected")
        self.velocities.linear.x = 0
        self.velocities.angular.z = 0
        time.sleep(3)
        self.object_handled = True

    def error(self):
        rospy.loginfo("Error: Unknown class")
        self.object_handled = True



    def __init__(self):
        # Params
        self.object_handled = rospy.get_param('/object_handled',True)
        self.velocities = Twist()
        self.velocities.linear.x = 0
        self.velocities.linear.y = 0
        self.velocities.linear.z = 0
        self.velocities.angular.x = 0
        self.velocities.angular.y = 0
        self.velocities.angular.z = 0
        

        self.options = {"stop":self.stop,
                        "turn_left":self.turn_left,
                        "go_straight":self.go_straight,
                        "round_about":self.round_about,
                        "construction_ahead":self.construction_ahead,
                        "turn_right":self.turn_right,
                        "give_way":self.give_way,
                        "error":self.error,}



        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(30)

        # Publishers
        self.pub = rospy.Publisher("/cmd_vel", Twist,queue_size=10)

        # Subscribers
        rospy.Subscriber("/sign",String,self.callback)

    def callback(self,msg):
        self.sign = msg
        #set object handling flag to object is currently handling
        self.object_handled = False
        rospy.set_param('/object_handled',self.object_handled)

        # handles object
        self.options.get(self.sign.data,"error")()

        # set object handling flag to object handled
        rospy.set_param('/object_handled',self.object_handled)

                
                
                

    def start(self):
        rospy.loginfo("Starting Sign Handling")
        # the while loop to publish and so the node does not shut down
        while not rospy.is_shutdown():
            if not self.object_handled:
                self.pub.publish(self.velocities)
                rospy.loginfo("handling object")
            self.loop_rate.sleep()



if __name__ == '__main__':
    try:
        rospy.init_node("line_detector", anonymous=True)
        my_node = Nodo()
        my_node.start()
    except rospy.ROSInterruptException:
        pass