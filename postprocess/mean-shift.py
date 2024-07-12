# -------------------------------------------------------------------------------
# Name:         show_M
# Description:
# Author:       Ming_King
# Date:         2024/2/29
# -------------------------------------------------------------------------------

import numpy as np
import cv2
from sklearn.cluster import MeanShift
import time
from test_Mutil import test


def mean_shift(out_file, if_single_img=True):
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

    start_time = time.time()

    # Clustering using the Mean Shift clustering algorithm
    # MeanShift(bandwidth=100) adjusts the bandwidth parameter as needed
    mean_shift = MeanShift(bandwidth=1)
    mean_shift.fit(masked_pixels)
    end_time = time.time()  # Record end time
    elapsed_time_ms = (end_time - start_time) * 1000  # The time taken to calculate (milliseconds)
    print("Elapsed time (milliseconds) mean-shift:", elapsed_time_ms)

    # Get the center point after clustering
    cluster_centers = mean_shift.cluster_centers_

    # Get the cluster to which each pixel belongs
    labels = mean_shift.labels_

    # Put the clustered pixels back into the masked area of ​​the original image
    result_image = np.zeros_like(image)
    for i, (row, col) in enumerate(zip(masked_pixels_indices[0], masked_pixels_indices[1])):
        if i < len(labels):  # Make sure the index is valid
            cluster_index = labels[i]
            for i in range(len(cluster_centers)):
                # Check if all three data in the sublist are less than 10
                if all(x < 20 for x in cluster_centers[i]):
                    # If all are less than 10, modify the sublist to [100, 100, 100]
                    cluster_centers[i] = [100, 100, 100]
            result_image[row, col] = cluster_centers[cluster_index]

    # Display the original image, mask image, and processed image
    cv2.imshow('Original Image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imshow('Mask', mask)
    cv2.imshow('Result Image', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    opening = cv2.morphologyEx(
        result_image, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8))
    # cv2.imwrite('D:/PythonProject/lanenet-lane-detection-pytorch-main/test_output/instance_image/279_E_M.png',
    #             cv2.cvtColor(opening, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_file, cv2.cvtColor(opening, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
