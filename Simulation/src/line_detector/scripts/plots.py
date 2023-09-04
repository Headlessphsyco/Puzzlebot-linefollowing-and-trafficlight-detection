#!/usr/bin/env python
import rospy
from line_detector.msg import line
from geometry_msgs.msg import Twist
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Node(object):
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(3, 1,1)
        self.ay = self.fig.add_subplot(3, 1,2)
        self.az = self.fig.add_subplot(3, 1,3)
        self.x = []
        self.y = []
        self.y1 = []
        self.y2 = []
        self.count = 0
        self.line = line()
        rospy.Subscriber("/cmd_vel",Twist,self.callback)


    def callback(self,msg):
        self.data = msg
        if self.count <1000:
            self.x.append(self.count)
            self.y.append(self.data.angular.x) #self.velocities.angular.x , self.velocities.angular.z, self.data.linear.x
            self.y1.append(self.data.linear.x)
            self.y2.append(self.data.angular.z)
            # Limit x and y lists to 20 items
            self.x = self.x[-1000:]
            self.y = self.y[-1000:]
            self.y1 = self.y1[-1000:]
            self.y2 = self.y2[-1000:]
        self.count += 1


    def start(self):
        rospy.loginfo("Starting Plotting")
        ani = animation.FuncAnimation(self.fig, self.animate, fargs=(self.x, self.y), interval=33)
        plt.show()
        rospy.spin()


    def animate(self, i, xs, ys):
        # Draw x and y lists
        self.ax.clear()
        self.ay.clear()
        self.az.clear()
        self.ax.plot(self.x, self.y,color = 'b')
        self.ay.plot(self.x, self.y1,color = 'b')
        self.az.plot(self.x, self.y2,color= 'b')
        self.ax.set_title('Error over time Kp = 0.4 Kd = 0')
        self.ay.set_title('linear velocity over time')
        self.az.set_title('angular velocity over time')
        # Format plot
        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.10)
        #plt.title('Error over time')
        self.ax.set_ylabel('Error')
        self.ay.set_ylabel('Linear Velocity (m/s)')
        self.az.set_ylabel('Angular Velocity (rads/s)')
        self.az.set_xlabel('Time Step')


if __name__ == '__main__':
    rospy.init_node("line_detector", anonymous=True)
    my_node = Node()
    my_node.start()

