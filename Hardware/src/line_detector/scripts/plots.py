#!/usr/bin/env python
import rospy
from line_detector.msg import line
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Node(object):
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.x = []
        self.y = []
        self.line = line()
        rospy.Subscriber("/lines",line,self.callback)


    def callback(self,msg):
        self.lines = msg
        etheta = (240 - self.lines.line_point_x[0])/240
        self.x.append(dt.datetime.now().strftime('%M:%S.%f'))
        self.y.append(etheta)

        # Limit x and y lists to 20 items
        self.x = self.x[-100:]
        self.y = self.y[-100:]


    def start(self):
        rospy.loginfo("Starting Plotting")
        ani = animation.FuncAnimation(self.fig, self.animate, fargs=(self.x, self.y), interval=33)
        plt.show()
        rospy.spin()


    def animate(self, i, xs, ys):
        # Draw x and y lists
        self.ax.clear()
        self.ax.plot(self.x, self.y)

        # Format plot
        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.30)
        plt.title('Error over time')
        plt.ylabel('error')


if __name__ == '__main__':
    rospy.init_node("line_detector", anonymous=True)
    my_node = Node()
    my_node.start()

