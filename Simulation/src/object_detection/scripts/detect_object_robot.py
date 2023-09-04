#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from object_detection.msg import detected_object
from object_detection.msg import detected_objects
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import imp
from cv_bridge import CvBridge

class Node(object):
    def __init__(self):
        # Define and parse input arguments
        parser = argparse.ArgumentParser()
        #parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
        #                    required=True)
        parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                            default='detect.tflite')
        parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                            default='labelmap.txt')
        parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                            default=0.5)
        parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                            default='1280x720')
        parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                            action='store_true')

        args, unknown = parser.parse_known_args()

        #MODEL_NAME = args.modeldir
        MODEL_NAME = "/home/headlessphsyco/OpenCV/tensor_model_lite/content/custom_model_lite"
        GRAPH_NAME = args.graph
        LABELMAP_NAME = args.labels
        self.min_conf_threshold = float(args.threshold)
        resW, resH = args.resolution.split('x')
        self.imW, self.imH = int(resW), int(resH)
        use_TPU = args.edgetpu


        # Import TensorFlow libraries
        # If using Coral Edge TPU, import the load_delegate library

        from tensorflow.lite.python.interpreter import Interpreter
        if use_TPU:
            from tensorflow.lite.python.interpreter import load_delegate

        # If using Edge TPU, assign filename for Edge TPU model
        if use_TPU:
            # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
            if (GRAPH_NAME == 'detect.tflite'):
                GRAPH_NAME = 'edgetpu.tflite'   

        # Path to .tflite file, which contains the model that is used for object detection
        PATH_TO_CKPT = os.path.join(MODEL_NAME,GRAPH_NAME)

        # Path to label map file
        PATH_TO_LABELS = os.path.join(MODEL_NAME,LABELMAP_NAME)

        # Load the label map
        with open(PATH_TO_LABELS, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        # First label is '???', which has to be removed.
        if self.labels[0] == '???':
            del(self.labels[0])

        # Load the Tensorflow Lite model.
        # If using Edge TPU, use special load_delegate argument
        if use_TPU:
            self.interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                    experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
            print(PATH_TO_CKPT)
        else:
            self.interpreter = Interpreter(model_path=PATH_TO_CKPT)

        self.interpreter.allocate_tensors()

        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

        self. floating_model = (self.input_details[0]['dtype'] == np.float32)

        self.input_mean = 127.5
        self.input_std = 127.5

        # Check output layer name to determine if this model was created with TF2 or TF1,
        # because outputs are ordered differently for TF2 and TF1 models
        outname = self.output_details[0]['name']

        if ('StatefulPartitionedCall' in outname): # This is a TF2 model
            self.boxes_idx, self.classes_idx, self.scores_idx = 1, 3, 0
        else: # This is a TF1 model
            self.boxes_idx, self.classes_idx, self.scores_idx = 0, 1, 2

        # Initialize frame rate calculation
        self.frame_rate_calc = 1
        self.freq = cv2.getTickFrequency()


        # Initializing node stuff
        # Initializing publisher
        self.pub = rospy.Publisher('/objects', detected_objects, queue_size=10)
        rospy.Subscriber("/video_source/raw",Image,self.live_sign_detection)
        # launching node
        self.rate = rospy.Rate(33) # 10hz
        self.br = CvBridge()


    def live_sign_detection(self,msg):
        
        # initializing variables
        Objects = detected_objects()
        #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
        # emptying objects data
        Objects.object= []

        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame1 = self.br.imgmsg_to_cv2(msg)

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if self.floating_model:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        # Perform the actual detection by running the model with the image as input
        self.interpreter.set_tensor(self.input_details[0]['index'],input_data)
        self.interpreter.invoke()

        # Retrieve detection results
        boxes = self.interpreter.get_tensor(self.output_details[self.boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
        classes = self.interpreter.get_tensor(self.output_details[self.classes_idx]['index'])[0] # Class index of detected objects
        scores = self.interpreter.get_tensor(self.output_details[self.scores_idx]['index'])[0] # Confidence of detected objects

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > self.min_conf_threshold) and (scores[i] <= 1.0)):
                # objected detected so 
                Object = detected_object()

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * self.imH)))
                xmin = int(max(1,(boxes[i][1] * self.imW)))
                ymax = int(min(self.imH,(boxes[i][2] * self.imH)))
                xmax = int(min(self.imW,(boxes[i][3] * self.imW)))
                
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = self.labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                Object.bounding_box = [ymin,xmin,ymax,xmax]
                Object.confidence = int(scores[i]*100)
                Object.class_ = object_name
                Object.classid = int(classes[i])
                Objects.object.append(Object)

            # Draw framerate in corner of frame
            cv2.putText(frame,'FPS: {0:.2f}'.format(self.frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

            # All the results have been drawn on the frame, so it's time to display it.
            cv2.imshow('Object detector', frame)
            cv2.waitKey(1)

            # Calculate framerate
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/self.freq
            self.frame_rate_calc= 1/time1

            # publsihing data
            self.pub.publish(Objects)


    def start(self):
        rospy.loginfo("Starting Object Detecting")
        # the while loop to publish and so the node does not shut down
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('detect_objects_robot', anonymous=False)
    my_node = Node()
    try:
        my_node.start()
    except rospy.ROSInterruptException:
        pass