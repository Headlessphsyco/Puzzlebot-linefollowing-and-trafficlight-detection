# import libraries of python OpenCV    
import cv2  
  
# import Numpy by alias name np  
import numpy as np  
  
# capture frames from a camera   
cap = cv2.VideoCapture(0)  
  
# loop runs if capturing has been initialized   
while (1):  
  
    # reads frames from a camera   
    ret, image = cap.read()  
  
    # to convert the image in grayscale  
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    threshold=250
    kernel_size = 15
    blur_gray = cv2.GaussianBlur(img,(kernel_size, kernel_size),0)
    ret, thresh1 = cv2.threshold(blur_gray, threshold, 255, cv2.THRESH_BINARY) 

    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0,255,0), 1)
    for i in range(len(contours)):
        cnt = contours[i]

        (x,y),radius = cv2.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        radius = int(radius)
        circle_image = cv2.circle(image,center,radius,(0,255,0),1)

    cv2.imshow("final circle",image)
  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# Close the window   
cap.release()  
  
# De-allocate any associated memory usage   
cv2.destroyAllWindows()  