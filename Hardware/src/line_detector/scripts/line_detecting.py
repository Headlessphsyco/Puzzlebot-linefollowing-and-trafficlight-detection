#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from line_detector.msg import line
import numpy as np
import cv2
from cv_bridge import CvBridge
from matplotlib import pyplot as plt
from operator import itemgetter

def calibrate(img):
	color = ('b','g','r')
	high_thresh_bgr = [0,0,0]
	for i,col in enumerate(color):
		hist = cv2.calcHist([img],[i],None,[256],[0,256])
		hist_kernel_size = 9
		hist = cv2.GaussianBlur(hist,(hist_kernel_size, hist_kernel_size),0)
		#plt.plot(hist,color=col)
		#plt.xlim([0,256])
		peak = []
		value = []
		for j in range(1,254,1):
			if hist[j]>hist[j-1] and hist[j]>hist[j+1]:
				value = [j,hist[j]]
				peak.append(value)
		peak = sorted(peak,key=itemgetter(1), reverse=True)
		high_thresh_bgr[i] = peak[1][0]+15

	img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	#plt.show()	

	high_thresh_hsv = [0,0,0]
	for i,col in enumerate(color):
		summed = 0
		hist = cv2.calcHist([img2],[i],None,[256],[0,256])
		hist_kernel_size = 9
		hist = cv2.GaussianBlur(hist,(hist_kernel_size, hist_kernel_size),0)
		#plt.plot(hist,color=col)
		#plt.xlim([0,256])
		peak = []
		value = []
		for j in range(1,254,1):
			if hist[j]>hist[j-1] and hist[j]>hist[j+1]:
				value = [j,hist[j]]
				peak.append(value)
		peak = sorted(peak,key=itemgetter(1), reverse=True)
		high_thresh_hsv[i] = peak[1][0]+15	
	#plt.show()

	return high_thresh_bgr,high_thresh_hsv

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

	
	high_count = 100
	self.low_b1 = np.uint8([high_count,high_count,high_count])	
	self.high_b1 = np.uint8([0,0,0])

	self.low_b = np.uint8([255,50,110])
	self.high_b = np.uint8([0,5,0])

	erod_size = 5
	self.erod_element = cv2.getStructuringElement(cv2.MORPH_RECT,(2*erod_size+1,2*erod_size+1),(erod_size,erod_size))

	dil_size = 20
	self.dil_element = cv2.getStructuringElement(cv2.MORPH_RECT,(2*dil_size+1,2*dil_size+1),(dil_size,dil_size))

	self.clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))

	self.show_image = rospy.get_param('/show_images',True)
	rospy.set_param('/calibrate',True)

        # Subscribers
        # rospy.Subscriber("/camera/image_raw",Image,self.callback) # for sim
        rospy.Subscriber("/video_source/raw",Image,self.callback) #for sim
        

        


    def callback(self, msg):
        #rospy.loginfo('Image received...')
	cal = rospy.get_param('/calibrate',True)
        # getting image
        self.image = self.br.imgmsg_to_cv2(msg,desired_encoding='passthrough')
        img = self.image
	# checking image size the result is (800 800 3)
        (row,col,chan)= img.shape

        # cropping image
        img = img[int(row*0.7):int(row*0.9), int(col/2-240):int(col/2+240)]
	
	kernel_size = 5        
	blur = cv2.GaussianBlur(img,(kernel_size, kernel_size),0)
	lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
	lab_planes = cv2.split(lab)
	lab_planes[0] = self.clahe.apply(lab_planes[0])
	lab = cv2.merge(lab_planes)
	clahe_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)        
	clahe_hsv = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2HSV)

	if cal == True:	
		rospy.loginfo('Calibrating')
		bgr,hsv = calibrate(clahe_bgr)
		gray = cv2.cvtColor(clahe_bgr,cv2.COLOR_BGR2GRAY) #making image grey
		self.ret_otsu, otsu = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

		rospy.loginfo(bgr)
		rospy.loginfo(hsv)
		rospy.loginfo(self.ret_otsu)

		#bgr = max(bgr)

		# In BGR
		self.low_b1 = np.uint8([bgr[0],bgr[1],bgr[2]])

		# in HSV
		self.low_b = np.uint8([255,100,hsv[2]])
		
		rospy.set_param('/calibrate',False)

	gray = cv2.cvtColor(clahe_bgr,cv2.COLOR_BGR2GRAY) #making image grey
	ret, otsu = cv2.threshold(gray,self.ret_otsu,255,cv2.THRESH_BINARY_INV)        

	# masking
	hsv_mask = cv2.inRange(clahe_hsv,self.high_b,self.low_b)
	bgr_mask = cv2.inRange(clahe_bgr,self.high_b1,self.low_b1)

	alpha = 0.5
	beta = 1-alpha

	mask1 = cv2.addWeighted(hsv_mask, alpha, bgr_mask, beta, 0)
	mask = cv2.addWeighted(mask1, alpha, otsu, beta, 0)

	ret, thresh = cv2.threshold(mask,130,255,cv2.THRESH_BINARY)
	processed = cv2.erode(thresh,self.erod_element)
	processed = cv2.dilate(processed,self.dil_element)

	contours,herarchy = cv2.findContours(processed,1,cv2.CHAIN_APPROX_NONE)
	if len(contours)>0:
		c=max(contours,key=cv2.contourArea)
		topmost = c[c[:,:,1].argmin()]
		bottemmost = c[c[:,:,1].argmax()]
		if topmost[0][1] < 72 and bottemmost[0][1] > 72:
			M= cv2.moments(c)
			if M['m00']!=0:
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])
				cv2.circle(img,(cx,cy),5,(255,255,255),-1)
				cv2.circle(img,(240,cy),5,(0,0,255),-1)
				self.lines.line_point_x= [cx]
				self.lines.line_point_y= [cy]
		cv2.drawContours(img,c,-1,(0,255,0),1)
	
	
        if self.show_image:
        	#cv2.imshow('CLAHE', clahe_bgr)
		#cv2.imshow('image', img)
		#cv2.imshow('HSV',hsv_mask)
		#cv2.imshow('BGR',bgr_mask)
		#cv2.imshow('mixed',mask)
		#cv2.imshow('mixed thresholded',thresh) 
		#cv2.imshow('after proccessing',processed)
		#cv2.imshow('Gray',otsu)
            	cv2.waitKey(1) #delay
	

	self.pub.publish(self.lines)
        #rospy.loginfo("Sending Lines")
        
        

    def start(self):
        rospy.loginfo("Starting Line detection")
        # the while loop to publish and so the node does not shut down
        rospy.spin()



if __name__ == '__main__':
    try:
        rospy.init_node("line_detector", anonymous=True)
        my_node = Nodo()
        my_node.start()
    except rospy.ROSInterruptException:
        rospy.loginfo("stopping line detector")
        cv2.destroyAllWindows()
        pass
