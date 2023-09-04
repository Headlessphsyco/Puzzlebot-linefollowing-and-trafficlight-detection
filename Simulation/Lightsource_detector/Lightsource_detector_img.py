import cv2  
import numpy as np  
  
# path to input image is specified and   
# image is loaded with imread command  
image = cv2.imread('/home/headlessphsyco/OpenCV/Lightsource_detector/test_images/3.png')

# to convert the image in grayscale  
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

threshold=240
kernel_size = 15
blur_gray = cv2.GaussianBlur(img,(kernel_size, kernel_size),0)
ret, thresh1 = cv2.threshold(blur_gray, threshold, 255, cv2.THRESH_BINARY) 

contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (0,255,0), 1)
cnt = contours[0]

x,y,w,h = cv2.boundingRect(cnt)
final_image=cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("orginal",image)
cv2.imshow("grey scale",img)
cv2.imshow("after threshhold",thresh1)
cv2.imshow("final circle",final_image)
# De-allocate any associated memory usage   
cv2.waitKey(0)
cv2.destroyAllWindows() 
