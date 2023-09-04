#!/usr/bin/env python
import numpy as np
import cv2
import sys
from cv_bridge import CvBridge
from matplotlib import pyplot as plt
from operator import itemgetter


def calibrate(img):
	total_count = img.shape[0]*img.shape[1]
	high_percentage = 80
	target_count = total_count*high_percentage*0.01
	print('total count = ',total_count)
	print('target_count = ',target_count)
	color = ('b','g','r')
	high_thresh_bgr = [0,0,0]
	for i,col in enumerate(color):
		summed = 0
		hist = cv2.calcHist([img],[i],None,[256],[0,256])
		plt.plot(hist,color=col)
		plt.xlim([0,256])
		for j in range(255,0,-1):
		  summed += int(hist[j])
		  if target_count <= summed:
			high_thresh_bgr[i] = j
			break
		  else:
			high_thresh_bgr[i] = 0

	img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	plt.show()	

	high_thresh_hsv = [0,0,0]
	for i,col in enumerate(color):
		summed = 0
		hist = cv2.calcHist([img2],[i],None,[256],[0,256])
		plt.plot(hist,color=col)
		plt.xlim([0,256])
		for j in range(255,0,-1):
		  summed += int(hist[j])
		  if target_count <= summed:
			high_thresh_hsv[i] = j
			break
		  else:
			high_thresh_hsv[i] = 0
	plt.show()

	return high_thresh_bgr,high_thresh_hsv

def calibrateTwo(img):
	color = ('b','g','r')
	high_thresh_bgr = [0,0,0]
	for i,col in enumerate(color):
		hist = cv2.calcHist([img],[i],None,[256],[0,256])
		hist_kernel_size = 15
		hist = cv2.GaussianBlur(hist,(hist_kernel_size, hist_kernel_size),0)
		plt.plot(hist,color=col)
		plt.xlim([0,256])
		peak = []
		value = []
		for j in range(1,254,1):
			if hist[j]>hist[j-1] and hist[j]>hist[j+1]:
				value = [j,hist[j]]
				peak.append(value)
		peak = sorted(peak,key=itemgetter(1), reverse=True)
		high_thresh_bgr[i] = peak[1][0]+15

	img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	plt.xlabel('Intensity')
	plt.ylabel('Count')
	plt.legend(['Blue Channel','Green Channel','Red Channel'])
	plt.show()	

	high_thresh_hsv = [0,0,0]
	for i,col in enumerate(color):
		summed = 0
		hist = cv2.calcHist([img2],[i],None,[256],[0,256])
		hist_kernel_size = 15
		hist = cv2.GaussianBlur(hist,(hist_kernel_size, hist_kernel_size),0)
		plt.plot(hist,color=col)
		plt.xlim([0,256])
		peak = []
		value = []
		for j in range(1,254,1):
			if hist[j]>hist[j-1] and hist[j]>hist[j+1]:
				value = [j,hist[j]]
				peak.append(value)
		peak = sorted(peak,key=itemgetter(1), reverse=True)
		high_thresh_hsv[i] = peak[1][0]+15
	plt.xlabel('Intensity')
	plt.ylabel('Count')
	plt.legend(['Hue Channel','Saturation Channel','Value Channel'])	
	plt.show()

	return high_thresh_bgr,high_thresh_hsv


cap = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=21/1 ! nvvidconv flip-method=rotate-180 ! video/x-raw, width=1280, height=720, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink')

w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
print('Src opened, %dx%d @ %d fps' % (w, h, fps))

bridge = CvBridge()

kernel_size = 5
show_image = True

erod_size = 5
erod_element = cv2.getStructuringElement(cv2.MORPH_RECT,(2*erod_size+1,2*erod_size+1),(erod_size,erod_size))

dil_size = 20
dil_element = cv2.getStructuringElement(cv2.MORPH_RECT,(2*dil_size+1,2*dil_size+1),(dil_size,dil_size))

for i in range(1,200,1):
	ret_val, img = cap.read()
ret_val, img = cap.read()
image = img
img = bridge.cv2_to_imgmsg(img,encoding='passthrough')
img = bridge.imgmsg_to_cv2(img,desired_encoding='passthrough')
# checking image size the result is (800 800 3)
(row,col,chan)= img.shape

# cropping image
img = img[int(row*0.8):int(row*1), int(col/2-240):int(col/2+240)]

blur = cv2.GaussianBlur(img,(kernel_size, kernel_size),0)
lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
lab_planes = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
lab_planes[0] = clahe.apply(lab_planes[0])
lab = cv2.merge(lab_planes)
clahe_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

gray = cv2.cvtColor(clahe_bgr,cv2.COLOR_BGR2GRAY) #making image grey
ret_otsu, otsu = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

hist = cv2.calcHist([gray],[0],None,[256],[0,256])
plt.plot(hist,color='black')
plt.xlim([0,256])
plt.xlabel('Intensity')
plt.ylabel('Count')
plt.legend(['Grey Channel'])
plt.show()

bgr,hsv = calibrateTwo(clahe_bgr)
print(bgr)
print(hsv)
print(ret_otsu)


# In BGR
high_count = 100
high_b1 = np.uint8([0,0,0])
#low_b1 = np.uint8([high_count,high_count,high_count])
low_b1 = np.uint8([bgr[0],bgr[1],bgr[2]])

# in HSV
#low_b = np.uint8([255,50,110])
low_b = np.uint8([255,100,hsv[2]])
high_b = np.uint8([0,5,0])

if cap.isOpened():
    while True:
        ret_val, img = cap.read()
	image = img
	img = bridge.cv2_to_imgmsg(img,encoding='passthrough')
	img = bridge.imgmsg_to_cv2(img,desired_encoding='passthrough')
	# checking image size the result is (800 800 3)
        (row,col,chan)= img.shape

        # cropping image
        imgs = image[int(row*0.8):int(row*1), int(col/2-240):int(col/2+240)]
	img = img[int(row*0.8):int(row*1), int(col/2-240):int(col/2+240)]
        
	blur = cv2.GaussianBlur(img,(kernel_size, kernel_size),0)
	lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
	lab_planes = cv2.split(lab)
	#clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
	lab_planes[0] = clahe.apply(lab_planes[0])
	lab = cv2.merge(lab_planes)
	clahe_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
	clahe_hsv = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2HSV)

	gray = cv2.cvtColor(clahe_bgr,cv2.COLOR_BGR2GRAY) #making image grey
	ret, otsu = cv2.threshold(gray,ret_otsu,255,cv2.THRESH_BINARY_INV)        

	# masking
	hsv_mask = cv2.inRange(clahe_hsv,high_b,low_b)
	bgr_mask = cv2.inRange(clahe_bgr,high_b1,low_b1)

	alpha = 0.5
	beta = 1-alpha

	mask1 = cv2.addWeighted(hsv_mask, alpha, bgr_mask, beta, 0)
	mask = cv2.addWeighted(mask1, 0.66, otsu, 0.34, 0)

	ret, thresh = cv2.threshold(mask,130,255,cv2.THRESH_BINARY)


	processed = cv2.erode(thresh,erod_element)
	processed = cv2.dilate(processed,dil_element)

	contours,herarchy = cv2.findContours(processed,1,cv2.CHAIN_APPROX_NONE)
	if len(contours)>0:
		c=max(contours,key=cv2.contourArea)
		topmost = c[c[:,:,1].argmin()]
		bottemmost = c[c[:,:,1].argmax()]
		#print(topmost[0][1])
		#print(bottemmost[0][1])
		if topmost[0][1] < 72 and bottemmost[0][1] > 72:
			M= cv2.moments(c)
			if M['m00']!=0:
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])
				cv2.circle(img,(cx,cy),5,(255,255,255),-1)	
		cv2.drawContours(img,c,-1,(0,255,0),1)
	
        
        if show_image:
        	cv2.imshow('CLAHE', clahe_bgr)
		cv2.imshow('image Final', img)
		cv2.imshow('image', imgs)
		cv2.imshow('HSV',hsv_mask)
		cv2.imshow('BGR',bgr_mask)
		cv2.imshow('mixed',mask)
		cv2.imshow('mixed thresholded',thresh) 
		cv2.imshow('after proccessing',processed)
		cv2.imshow('Gray',otsu)
        if cv2.waitKey(1) & 0xFF == ord('q'):
        	break
else:
    print("pipeline open failed")

print("successfully exit")
cap.release()
out.release()
# Destroy all the windows
cv2.destroyAllWindows()
