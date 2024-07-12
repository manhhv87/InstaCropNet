# -------------------------------------------------------------------------------
# Name:         show_K2
# Description:
# Author:       Ming_King
# Date:         2024/2/29
# -------------------------------------------------------------------------------

import numpy as np
import cv2
from sklearn.cluster import KMeans
import time
from test_Mutil import test


def k_means(out_file, if_single_img=True):
    # Reading an Image
    # image = cv2.imread(r'E:\A_trans\results\AH\3\Unet\279_instance_output.png')
    # mask = cv2.imread(r'E:\A_trans\results\AH\3\Deeplabv3+\279_binary_output.png', cv2.IMREAD_GRAYSCALE)
    f1, f2, f3 = test(if_single_img)
    image = cv2.imread(f2)
    mask = cv2.imread(f3, cv2.IMREAD_GRAYSCALE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            kernel=np.ones((5, 5), np.uint8))

    # Convert the image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask[mask > 50] = 255
    mask[mask <= 50] = 0

    # Extract the non-zero pixels in the second image that correspond to the mask
    masked_pixels = image[np.where(mask != 0)]

    # Specify the number of clusters to be divided into
    num_clusters = 4

    start_time = time.time()  # Recording start time

    # Clustering using the K-Means clustering algorithm
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(masked_pixels)

    end_time = time.time()  # Record end time

    # The time taken to calculate (milliseconds)
    elapsed_time_ms = (end_time - start_time) * 1000
    print("Time consumed (milliseconds) K-means:", elapsed_time_ms)

    # Get cluster centers
    cluster_centers = kmeans.cluster_centers_

    # Get the cluster to which each pixel belongs
    labels = kmeans.labels_

    # Put the clustered pixels back into the masked area of ​​the original image
    result_image = np.zeros_like(image)
    # result_image = np.ones_like(image)
    non_zero_indices = np.where(mask != 0)
    for i in range(len(non_zero_indices[0])):
        row = non_zero_indices[0][i]
        col = non_zero_indices[1][i]
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
    # cv2.imwrite('D:/PythonProject/lanenet-lane-detection-pytorch-main/test_output/instance_image/279_M.png', mask)
    opening = cv2.morphologyEx(
        result_image, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))
    # cv2.imwrite('D:/PythonProject/lanenet-lane-detection-pytorch-main/test_output/instance_image/279_U.png',
    #             cv2.cvtColor(opening, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_file, cv2.cvtColor(opening, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
