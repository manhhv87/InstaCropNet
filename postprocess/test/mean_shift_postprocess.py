import numpy as np
from sklearn.cluster import MeanShift
import cv2
import matplotlib.pyplot as plt


def post_process_clusters(labels, min_cluster_size):
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    # Mark clusters smaller than the minimum cluster size as noise
    noise_labels = unique_labels[label_counts < min_cluster_size]

    # Mark noise points as zero
    for noise_label in noise_labels:
        labels[labels == noise_label] = 0
    return labels


def lane_detection(binary_image_path, instance_image_path, delta_v, min_cluster_size=100):
    # Read binary image and feature vector image
    binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)

    # Set a threshold value, set pixels greater than the threshold value to white, 
    # and pixels less than or equal to the threshold value to black
    _, binary_smoothed = cv2.threshold(
        binary_image, 200, 255, cv2.THRESH_BINARY)

    # Use median filtering for smoothing
    binary_smoothed = cv2.medianBlur(binary_smoothed, 5)
    instance_image = cv2.imread(instance_image_path)

    # Get the coordinates of non-zero pixels
    non_zero_points = np.column_stack(np.where(binary_smoothed > 0))

    # Get the feature vector (color value)
    features = [instance_image[point[0], point[1]]
                for point in non_zero_points]

    # Convert to numpy array
    features = np.array(features)

    # Clustering using MeanShift
    clustering = MeanShift(bandwidth=delta_v)
    clustering.fit(features)

    # Get cluster centers and labels
    cluster_centers = clustering.cluster_centers_
    labels = clustering.labels_

    # Create a lane assignment dictionary
    lanes = {label: [] for label in set(labels)}

    # Post-processing: Removing small clusters
    # labels = post_process_clusters(labels, min_cluster_size)

    # Pixel allocation based on clustering results
    for i, point in enumerate(non_zero_points):
        label = labels[i]
        lanes[label].append(point)

    # Example result image after creating clustering
    instance_result = np.zeros_like(binary_image)

    # Pixel marking based on lane line assignment results
    for label, lane_points in lanes.items():
        for point in lane_points:
            instance_result[point[0], point[1]] = label + 1

    # Save the instance result image
    # cv2.imwrite("instance_result.jpg", instance_result)

    # Display example result image
    plt.imshow(instance_result, cmap='jet')  # Use jet colormap to show different instances
    plt.title('Instance Segmentation Result')
    plt.colorbar()
    plt.show()


# Example call
binary_image_path = 'test_output/2_binary_image.jpg'
instance_image_path = 'test_output/2_instance_image.jpg'
delta_v = 50  # Your radius parameter

lane_detection(binary_image_path, instance_image_path, delta_v)
