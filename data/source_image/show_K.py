import numpy as np
import cv2
from sklearn.cluster import KMeans

# Reading an Image
image = cv2.imread(
    r'D:\PythonProject\lanenet-lane-detection-pytorch-main\test_output\1_instance_image.jpg')

# Convert the image to RGB format
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get the shape of an image
height, width, channels = image.shape

# Convert image data into a 2D array
image_data = image.reshape((-1, 3))

# Specify the number of clusters to be divided into
num_clusters = 5

# Clustering using the K-Means clustering algorithm
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(image_data)

# Get cluster centers
cluster_centers = kmeans.cluster_centers_

# Get the cluster to which each pixel belongs
labels = kmeans.labels_

# Create a new image where each pixel's color is determined by the center color of the cluster it belongs to.
clustered_image = np.zeros_like(image)
for i in range(height):
    for j in range(width):
        cluster_index = labels[i * width + j]
        clustered_image[i, j] = cluster_centers[cluster_index]

# Display the original image and the clustered image
cv2.imshow('Original Image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.imshow('Clustered Image', cv2.cvtColor(clustered_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
