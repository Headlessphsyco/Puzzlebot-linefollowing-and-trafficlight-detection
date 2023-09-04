#!/usr/bin/env python
import rospy
from object_detection.msg import detected_object
from object_detection.msg import detected_objects
from std_msgs.msg import String
import numpy as np
import time
import math


class Nodo(object):
    def update_array(self,saved_array, new_array, consecutive_count, temp_array):
        updated_array = saved_array

        for j,line in enumerate(new_array):
            line_found = False

            for i, saved_line in enumerate(saved_array):
                if all(abs(num - saved_num) <= 100 for num, saved_num in zip(line, saved_line)):
                    # Line is within range of saved line, overwrite it
                    updated_array[i] = line
                    consecutive_count[i] = 0
                    line_found = True
                    break

            if not line_found:
                # Line is not within range of any saved line, check temp array
                for i, temp_line in enumerate(temp_array):
                    if all(abs(num - temp_num) <= 100 for num, temp_num in zip(line, temp_line)):
                        # Line is within range of temp line, overwrite it
                        temp_array[i] = line
                        #consecutive_count[i + len(saved_array)] += 1
                        line_found = True
                        break

                if not line_found:
                    # Line is not within range of any temp line, add it to temp array
                    temp_array.append(line)
                    self.temp_sign.append(self.class_[j])
                    consecutive_count.append(0)

        # Check consecutive counts and move lines to saved array if necessary
        lines_to_remove = []
        lines_added = 0
        for i, count in enumerate(consecutive_count):
            if count >= 5:
                if i < len(saved_array):
                    del updated_array[i]
                    del self.handled[i]
                    del self.sign[i]
                else:
                    updated_array.append(temp_array[i - len(saved_array)])
                    self.handled.append(False)
                    self.sign.append(self.temp_sign[i - len(saved_array)])
                    lines_added += 1
                    del temp_array[i - len(saved_array)]
                    del self.temp_sign[i - len(saved_array)]
                del consecutive_count[i]
            else:
                consecutive_count[i] += 1
                

        # inseting a consecutive count for lines that were added from temporary array to the updated array
        if not (lines_added == 0):
            for i in range(lines_added):
                consecutive_count.insert(0,len(saved_array))

        # updating the global arrays
        return updated_array, temp_array



    def __init__(self):
        # Params
        self.Objects = detected_objects()
        self.Object = detected_object()
        self.object_handled = rospy.get_param('/object_handled',True)
        self.saved_array = []
        self.consecutive_count = [0] * len(self.saved_array)
        self.temp_array = []
        self.handled = []
        self.sign = []
        self.temp_sign = []
        self.signHandling = String()

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(30)

        # Publishers
        self.pub = rospy.Publisher("/sign", String,queue_size=10)

        # Subscribers
        rospy.Subscriber("/objects",detected_objects,self.callback)

    def callback(self,msg):
        self.object_detected = rospy.get_param('/object_detected',False)
        self.object_handled = rospy.get_param('/object_handled',True)
        self.Objects = msg
        self.class_=[]
        bounding_box = []
        if len(self.Objects.object) >0:
            for Objects in self.Objects.object:
                self.class_.append(Objects.class_)
                bounding_box.append(Objects.bounding_box)
            self.saved_array,self.temp_array= self.update_array(self.saved_array,bounding_box,self.consecutive_count,self.temp_array)
        else:
            self.saved_array,self.temp_array= self.update_array(self.saved_array,bounding_box,self.consecutive_count,self.temp_array)
        #rospy.loginfo(bounding_box)
        rospy.loginfo(self.saved_array)
        rospy.loginfo(self.temp_array)
        rospy.loginfo(self.consecutive_count)
        rospy.loginfo(self.sign)

        if (self.object_handled):
            for i,box in enumerate(self.saved_array):
                if abs(box[0]-box[2])>= 100 and abs(box[1]-box[3])>= 100 and not self.handled[i]:
                        self.object_detected = True
                        rospy.set_param('/object_handled',False)
                        self.signHandling.data = self.sign[i]
                        self.pub.publish(self.signHandling)
                        self.handled[i] = True
                        break
                else:
                    self.object_detected = False
        else:
            self.object_detected = False

        rospy.set_param('/object_detected',self.object_detected)

                
                
                

    def start(self):
        rospy.loginfo("Starting Sign Handling")
        # the while loop to publish and so the node does not shut down
        while not rospy.is_shutdown():
            self.loop_rate.sleep()



if __name__ == '__main__':
    try:
        rospy.init_node("Sign_tracker", anonymous=False)
        my_node = Nodo()
        my_node.start()
    except rospy.ROSInterruptException:
        pass