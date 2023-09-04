#!/usr/bin/env python
import rospy
from object_detection.msg import detected_object
from object_detection.msg import detected_objects
from std_msgs.msg import Bool
import numpy as np

def simulate_object():
    pub = rospy.Publisher('objects', detected_objects, queue_size=10)
    pub2 = rospy.Publisher('object_detected',Bool,queue_size=10)
    rospy.init_node('simulate_detected_object', anonymous=True)
    sign = rospy.get_param("/sign","stop")
    Object = detected_object()
    Objects = detected_objects()
    Object.class_ = sign
    Object.classid = 0
    Object.confidence = 70
    Object.bounding_box = [400,400,100,100]
    Objects.object.append(Object)
    Object_detected = Bool()
    Object_detected.data = True
    rospy.loginfo("publising: ")
    rospy.loginfo(Objects)
    rospy.loginfo(Object_detected)
    pub.publish(Objects)
    pub2.publish(Object_detected)




if __name__ == '__main__':
    try:
        simulate_object()
    except rospy.ROSInterruptException:
        pass