# import libraries of python OpenCV    
import cv2  
  
# import Numpy by alias name np  
import numpy as np  

import skimage.exposure as exposure

def get_mid_arr(arr: np.ndarray, k: int) -> np.ndarray:
    mid_arr = arr.copy()
    upper = np.triu_indices(mid_arr.shape[0], k=k)
    mid_arr[upper] = 0
    lower = np.tril_indices(mid_arr.shape[0], k=-k)
    mid_arr[lower] = 0
    return mid_arr


def get_upper_arr(arr, k: int) -> np.ndarray:
    upper_arr = arr.copy()
    lower_triangle_indices = np.tril_indices(upper_arr.shape[0], k= k - 1)

    upper_arr[lower_triangle_indices] = 0

    return upper_arr


def get_lower_arr(arr, k: int) -> np.ndarray:
    lower_arr = arr.copy()
    upper_triangle_indices = np.triu_indices(lower_arr.shape[0], k = - k + 1)
    lower_arr[upper_triangle_indices] = 0
    return lower_arr


class TraficLightHistogramClassifier:
    
    def __init__(
        self,
        hist_cutting_treshold=55,  # optimal value found for my dataset
        probability_boundary=0.09,  # optimal value found for my dataset
    ) -> None:
        self.hist_cutting_treshold = hist_cutting_treshold
        self.probability_boundary = probability_boundary

    
    def predict(self, img):
        # calculate 2D histograms for pairs of channels: GR
        hist = cv2.calcHist([img], [1, 2], None, [256, 256], [0, 256, 0, 256])
        # hist is float and counts need to be scale to range 0 to 255
        scaled_hist = (
            exposure.rescale_intensity(hist, in_range=(0, 1), out_range=(0, 255))
            .clip(0, 255)
            .astype(np.uint8)
        )

        # Split histogram into 3 regions
        (yellow_region, green_region, red_region) = (
            get_mid_arr(scaled_hist, self.hist_cutting_treshold),
            get_lower_arr(scaled_hist, self.hist_cutting_treshold),
            get_upper_arr(scaled_hist, self.hist_cutting_treshold),
        )
        

        # Count how many non zero values in each region
        (red_count, green_count, yellow_count) = (
            np.count_nonzero(red_region),
            np.count_nonzero(green_region),
            np.count_nonzero(yellow_region),
        )

        # Calculate total non-zero values
        total_count = red_count + green_count + yellow_count
        if total_count == 0:
            total_count = 1

        # Calculate red and green percentage
        red_percentage, green_percentage = (
            red_count / total_count,
            green_count / total_count,
        )

        # Logic for deciding color
        if green_percentage > self.probability_boundary:
            predict = "green"
        elif red_percentage > self.probability_boundary:
            predict = "red"
        else:
            predict = "yellow"


        return predict
  
# capture frames from a camera   
cap = cv2.VideoCapture(0)  
  
# Create classifier
clf = TraficLightHistogramClassifier()
# loop runs if capturing has been initialized   
while (1):  
  
    # reads frames from a camera   
    ret, image = cap.read()  
  
    # to convert the image in grayscale  
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    threshold=250
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(img,(kernel_size, kernel_size),0)
    ret, thresh1 = cv2.threshold(blur_gray, threshold, 255, cv2.THRESH_BINARY) 

    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0,255,0), 1)
    for i in range(len(contours)):
        cnt = contours[i]
        x,y,w,h = cv2.boundingRect(cnt)
        detected_light = image[int(x-(w+10)/2):int(x+(w+10)/2),int(y-(h+10)/2):int(y+(h+10)/2)]
        #cv2.imshow("testing",detected_light)
        predicted_color = clf.predict(detected_light)
        if predicted_color == "red":
            image=cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
        elif predicted_color == "green":
            image=cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        else:
            image=cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)

    cv2.imshow("final circle",image)
    cv2.imshow("thresholded",thresh1)
    cv2.imshow("grey",img)
  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# Close the window   
cap.release()  
  
# De-allocate any associated memory usage   
cv2.destroyAllWindows()  