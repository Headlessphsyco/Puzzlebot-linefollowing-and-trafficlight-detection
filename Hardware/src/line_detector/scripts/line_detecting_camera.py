#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from line_detector.msg import line
import numpy as np
import cv2
from cv_bridge import CvBridge


class Nodo(object):
    def __init__(self):
        # Params
        self.image = None
        self.br = CvBridge()
        self.lines = line()
        self.lines.line_point_x = [245,240,240,240,235]
        self.lines.line_point_y = [100,150,200,250,300]
        self.coefficients = [1,1]
        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(30)

        # Publishers
        self.pub = rospy.Publisher('/lines', line,queue_size=10)

        # Subscribers
        # rospy.Subscriber("/camera/image_raw",Image,self.callback) # for sim
        rospy.Subscriber("/video_source/raw",Image,self.callback) #for sim
        

        


    def callback(self, msg):
        rospy.loginfo('Image received...')
        # getting image
        self.image = self.br.imgmsg_to_cv2(msg)
        image = self.image

        # checking image size the result is (800 800 3)
        (row,col,chan)= self.image.shape

        

        # cropping image
        self.image = self.image[int(row*0.45):int(row*1), int(col/2-240):int(col/2+240)]
        
        
        #Method 1
	#alpha = rospy.get_param('/alpha', 2)
	#beta = rospy.get_param('/beta', 10)
	#contrasted = cv2.convertScaleAbs(self.image,alpha=alpha,beta=beta) #adding contrast to image	

	
	# grey scaling and blurring        
	#gray = cv2.cvtColor(contrasted,cv2.COLOR_BGR2GRAY) #making image grey

		
	
        #kernel_size = rospy.get_param('/kernel_size', 5)
        #blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0) # bluring image
        
		
	#high_percentage = rospy.get_param('/high_percentage', 75)
	#hist = cv2.calcHist([blur_gray], [0],None,[256],[0,256]).flatten()
	#total_count = blur_gray.shape[0]*blur_gray.shape[1]
	#target_count = total_count*high_percentage*0.01
	#summed = 0
	#for i in range(255,0,-1):
	#   summed += int(hist[i])
	#  if target_count <= summed:
	#	high_thresh = i
	#	break
	#   else:
	#	high_thresh = 0

	#thresh = rospy.get_param('/binary_threshold', 200)# performing binary threshold
        #ret, thresh2 = cv2.threshold(blur_gray, high_thresh, 255, cv2.THRESH_BINARY) 
        
	# Method 2
	high_count = rospy.get_param('/high_count', 100)
	high_b = np.uint8([0,0,0])
	low_b = np.uint8([high_count,high_count,high_count])
	mask = cv2.inRange(self.image,high_b,low_b)

        # doing canny edge detection
        low_threshold = rospy.get_param('/edge_low_threshold', 50)
        high_threshold = rospy.get_param('/edge_high_threshold', 150)
        edges = cv2.Canny(mask, low_threshold, high_threshold)
        

        # setting up parameters for Hough on edge detection
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = rospy.get_param('/line_threshold', 25)  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = rospy.get_param('/min_line_length', 20)  # minimum number of pixels making up a line
        max_line_gap = rospy.get_param('/max_line_gap', 20)  # maximum gap in pixels between connectable line segments
        line_image = np.copy(self.image) * 0  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

        # drawing lines for each line detected also getting slope and intercept of each line
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)

            lines = lines[:, 0, :]  # Flatten the lines array
            x1_values, y1_values, x2_values, y2_values = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]
            
            # drawing lines for each line detected also getting slope and intercept of each line
            slope = (y2_values - y1_values).astype(float) / (x2_values - x1_values)
            y_intercept = y1_values - slope * x1_values

            # setting pixel values of checking line at
            y_values = [100,150,200,250,300]
            x_values = get_x_values(slope,y_intercept,y_values, y1_values, y2_values)

            # getting midpoints for lines close to each other
            high_range = rospy.get_param('/midpoint_high', 150)
            low_range = rospy.get_param('/midpoint_low', 30)
            midpoints = []
            # for y in x_values:
            #     midpoints.append(isolate_and_get_midpoint(y,low_range,high_range))
            for x in x_values:
                line_midpoints = isolate_and_get_midpoint(x, int(low_range), int(high_range))  # *((1/len(x_values))*i)
                if line_midpoints is not None:  # Check if line_midpoints is valid
                    midpoints.append(line_midpoints)

            

            # drawing circles for the mid points and also for the buggy path going straigh
            average_midpoint = []
            for points in midpoints:
                average_point = np.mean(points)
                average_midpoint.append(average_point)
            
            
        else:
            y_values = self.lines.line_point_y
            average_midpoint = self.lines.line_point_x

        i= 0
        for point in average_midpoint:
            y = y_values[i]
            cv2.circle(line_image,(240,y),5,(0,0,255),-1)
            if not np.isnan(point):
                cv2.circle(line_image,(int(point),y),5,(0,255,255),-1)
            i= i+1
    
        # generating mast to remove nan values
        nan_mask = np.isnan(average_midpoint) | np.isnan(y_values)
        valid_indices = np.logical_not(nan_mask)
        average_midpoint = np.array(average_midpoint)
        y_values = np.array(y_values)
        average_midpoint_valid = average_midpoint[valid_indices]
        y_values_valid = y_values[valid_indices]

        # generating message to publish    cv2.imshow('after image processing', thresh2) 
        if not np.all(np.isnan(average_midpoint)):
            self.lines.line_point_x = average_midpoint_valid
            self.lines.line_point_y = y_values_valid
            coefficients = np.polyfit(average_midpoint_valid, y_values_valid, 1)
            self.coefficients = coefficients
        self.lines.slope = self.coefficients[0]
        self.lines.intercept = self.coefficients[1]

        if not np.isnan(self.coefficients[0]) and not np.isnan(self.coefficients[1]):
            x_start = int((1 - self.coefficients[1]) / self.coefficients[0])
            x_end = int((400 - self.coefficients[1]) / self.coefficients[0])
            cv2.line(line_image, (x_start, 1), (x_end, 400), (0, 255, 255), 1)


        # Draw the lines on the  image
        lines_edges = cv2.addWeighted(self.image, 0.8, line_image, 1, 0)
        show_image = rospy.get_param('/show_images',False)
        if show_image:
            #cv2.imshow('image', image)
	    #cv2.imshow('after image contrast', contrasted) 
            #cv2.imshow('method 1', thresh2)
	    #cv2.imshow('method 2',mask) 
            #cv2.imshow('edge detection',edges)
            #cv2.imshow('image with line', lines_edges)
            #cv2.imshow('lines', line_image)
            cv2.waitKey(1) #delay
        
        

    def start(self):
        rospy.loginfo("Starting Line detection")
        # the while loop to publish and so the node does not shut down
        while not rospy.is_shutdown():
            if self.image is not None:
                self.pub.publish(self.lines)
                rospy.loginfo("Sending Lines")
            self.loop_rate.sleep()


def get_x_values(slopes, y_intercepts, y_values, y1, y2):
    x_values = []

    for y in y_values:
        x_line = []
        for i in range(len(slopes)):
            if slopes[i] != 0 and (min(y1[i],y2[i])< y < max(y1[i],y2[i])):
                x = (y - y_intercepts[i]) / slopes[i]
            else:
                # Handle the case when the line is vertical (infinite x-values)
                x = float('inf')
            x_line.append(x)
        x_values.append(x_line)

    return x_values

def isolate_and_get_midpoint(numbers, lower_range, upper_range):
    isolated_numbers = []
    
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if 0<= numbers[i] <=480:
                difference = abs(numbers[i] - numbers[j])
                if lower_range <= difference <= upper_range:
                    isolated_numbers.append((numbers[i] + numbers[j]) / 2)
    
    return isolated_numbers
    


if __name__ == '__main__':
    try:
        rospy.init_node("line_detector", anonymous=True)
        my_node = Nodo()
        my_node.start()
    except rospy.ROSInterruptException:
        rospy.loginfo("stopping line detector")
        cv2.destroyAllWindows()
        pass
