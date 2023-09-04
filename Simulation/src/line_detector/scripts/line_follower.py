#!/usr/bin/env python
import rospy
from line_detector.msg import line
from geometry_msgs.msg import Twist
import numpy as np
import time

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_time = 0
        self.prev_error = 0
        self.integral = 0

    def compute(self, error):
        # Calculate time difference since the last computation
        current_time = time.time()
        dt = current_time - self.prev_time

        delta_error = error - self.prev_error

        # Proportional term
        proportional = self.Kp * error

        # Integral term
        self.integral += self.Ki * error * dt

        # Derivative term
        derivative = self.Kd * delta_error / dt

        # Compute the output
        output = proportional + self.integral + derivative

        # Update the previous values for the next iteration
        self.prev_time = current_time
        self.prev_error = error

        return output
    
    def update(self,Kp,Ki,Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd


class Node(object):
    def __init__(self):
        #initliaizing messeages and variables
        self.lines = line()
        self.velocities = Twist()
        self.velocities.linear.x = 0
        self.velocities.linear.y = 0
        self.velocities.linear.z = 0
        self.velocities.angular.x = 0
        self.velocities.angular.y = 0
        self.velocities.angular.z = 0
        self.object_detect = rospy.get_param('/object_handled',True)


        KPtheta = rospy.get_param('/kptheta',0.5)
        KItheta = rospy.get_param('/kitheta',0)
        KDtheta = rospy.get_param('/kdtheta',0.1)

        self.pid = PIDController(KPtheta,KItheta,KDtheta)

        self.loop_rate = rospy.Rate(30)
        

        # Publishers
        self.pub = rospy.Publisher("/cmd_vel", Twist,queue_size=10)

        # Subscribers
        rospy.Subscriber("/lines",line,self.callback)


    def callback(self, msg):
        self.object_detect = rospy.get_param('/object_handled',True)
        if self.object_detect:
            rospy.loginfo("Recived Lines...")

            self.lines = msg

            # line following code
            etheta = (240 - self.lines.line_point_x[0])/240
            max_speed = rospy.get_param('/max_speed',0.3)
            self.velocities.linear.x = max_speed*(1-abs(etheta))
            self.velocities.angular.z = self.pid.compute(etheta)
            self.velocities.angular.x = etheta

            KPtheta = rospy.get_param('/kptheta',0.5)
            KItheta = rospy.get_param('/kitheta',0)
            KDtheta = rospy.get_param('/kdtheta',0.1)

            self.pid.update(KPtheta,KItheta,KDtheta)

      
                

    def start(self):
        rospy.loginfo("Starting Line detection")
        # the while loop to publish and so the node does not shut down
        while not rospy.is_shutdown():
            if self.lines is not None:
                self.pub.publish(self.velocities)
                rospy.loginfo("Sending velocities")
            self.loop_rate.sleep()

    def end(self):
        self.velocities.linear.x = 0
        self.velocities.angular.z = 0
        self.pub.publish(self.velocities)
        self.loop_rate.sleep()
        rospy.loginfo("Stopping line following")






if __name__ == '__main__':
    rospy.init_node("line_follower", anonymous=False)
    my_node = Node()
    try:
        my_node.start()
    except rospy.ROSInterruptException:
        my_node.end()
        pass