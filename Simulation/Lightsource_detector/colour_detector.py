# Dependencies
import cv2
import numpy as np
import skimage.exposure as exposure
from matplotlib import pyplot as plt

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
        # Create empty list for holding predictions
        predictions = []
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

        # Append to predictions
        predictions.append(predict)

        
        color = ('b','g','r')
        for i,col in enumerate(color):
            hist = cv2.calcHist([img],[i],None,[256],[0,256])
            plt.plot(hist,color = col)
            plt.xlim([0,256])
        plt.show()
        cv2.imshow("hist", scaled_hist)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return predictions


def main():

    # Create classifier
    clf = TraficLightHistogramClassifier()

    img = cv2.imread("/home/headlessphsyco/OpenCV/Lightsource_detector/test_images/2.jpg")

    # "Predict" lights
    y_pred = clf.predict(img)

    # Print predict "labels"
    print(y_pred)


if __name__ == "__main__":
    main()
