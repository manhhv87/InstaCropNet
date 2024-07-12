# -------------------------------------------------------------------------------
# Name:         show_B
# Description:
# Author:       Ming_King
# Date:         2024/2/29
# -------------------------------------------------------------------------------
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import time
from test_Mutil import test


def DBSCAN(out_file, if_single_img=True):
    # Reading an Image
    # image = cv2.imread(r'E:\A_trans\results\AH\3\Enet\279_instance_output.png')
    # mask = cv2.imread(r'E:\A_trans\results\AH\3\Enet\279_binary_output.png', cv2.IMREAD_GRAYSCALE)
    f1, f2, f3 = test(if_single_img)
    image = cv2.imread(f2)
    mask = cv2.imread(f3, cv2.IMREAD_GRAYSCALE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            kernel=np.ones((5, 5), np.uint8))
    
    # Convert the image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extract non-zero pixels with pixel values ​​greater than 100 in the second image
    threshold_value = 100
    masked_pixels_indices = np.where(mask > threshold_value)
    masked_pixels = image[masked_pixels_indices]
    # masked_pixels = masked_pixels[np.where(masked_pixels[:, 0] > threshold_value)]

    start_time = time.time()

    # Clustering using the DBSCAN clustering algorithm
    dbscan = DBSCAN()  # Use default parameters or DBSCAN(eps=100, min_samples=50) Adjust eps and min_samples parameters as needed
    dbscan.fit(masked_pixels)
    end_time = time.time()  # Record end time
    elapsed_time_ms = (end_time - start_time) * 1000  # The time taken to calculate (milliseconds)
    print("Elapsed time (milliseconds) DBSCAN:", elapsed_time_ms)

    # Get the cluster to which each pixel belongs
    labels = dbscan.labels_

    # Get the number of clusters after clustering
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Put the clustered pixels back into the masked area of ​​the original image
    result_image = np.zeros_like(image)
    for i, (row, col) in enumerate(zip(masked_pixels_indices[0], masked_pixels_indices[1])):
        if i < len(labels):  # Make sure the index is valid
            cluster_index = labels[i]
            if cluster_index != -1:  # If it weren't for the noise
                result_image[row, col] = masked_pixels[i]

    # Display the original image, mask image, and processed image
    cv2.imshow('Original Image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imshow('Mask', mask)
    cv2.imshow('Result Image', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    opening = cv2.morphologyEx(
        result_image, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))
    # cv2.imwrite('D:/PythonProject/lanenet-lane-detection-pytorch-main/test_output/instance_image/279_E_B.png',
    #             cv2.cvtColor(opening, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_file, cv2.cvtColor(opening, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
