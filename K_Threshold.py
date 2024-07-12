import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import cv2

# # Generate some sample data
# np.random.seed(0)
# X = 2 * np.random.rand(100, 1)
# y = 4 + 3 * X + np.random.randn(100, 1)
#
# # Linear regression using the least squares method
# model = LinearRegression()
# model.fit(X, y)
#
# # Get the slope and intercept of the fitted line
# slope = model.coef_[0].item()
# intercept = model.intercept_.item()
#
# # Plotting the original data and the fitted line using matplotlib
# plt.scatter(X, y, label='Original Data')
# plt.plot(X, model.predict(X), color='red', label='Fitted Line (y = {:.2f}x + {:.2f})'.format(slope, intercept))
# # plt.legend()
# # plt.xlabel('X')
# # plt.ylabel('y')
# # plt.title('Linear Regression using Least Squares Method')
# plt.show()


# # Function to fit a straight line using least squares
# def fit_line(x, y):
#     mask = ~np.isnan(x) & ~np.isnan(y)  # Create a mask to filter out NaN values
#     x, y = x[mask], y[mask]
#
#     if np.all(x == x[0]):  # Check if x-values are constant
#         m = np.inf  # Slope is set to infinity for constant x-values
#         c = np.mean(y)  # Intercept is the mean of y-values
#     else:
#         A = np.vstack([x, np.ones_like(x)]).T
#         m, c = np.linalg.lstsq(A, y, rcond=None)[0]
#     return m, c
#
#
# # Function to plot data points and fitted line on an image
# def plot_data_and_line(image, data_points, colors):
#     plt.imshow(image)  # Display the image
#
#     for i, (x, y) in enumerate(data_points):
#         m, c = fit_line(x, y)
#         if np.isinf(m):
#             plt.axvline(x[0], color=colors[i], label=f'Line {i + 1}')
#         else:
#             plt.plot(x, m * x + c, color=colors[i], label=f'Line {i + 1}')
#         plt.scatter(x, y, color=colors[i])  # Scatter plot for data points
#
#     # Set axis limits for lower left corner as origin
#     plt.xlim((0, image.shape[1]))
#     plt.ylim((0, image.shape[0]))
#
#     plt.legend()
#     plt.show()
#
#
# # Example usage
# if __name__ == "__main__":
#     # Replace this with your actual image and data points
#     image = np.zeros((10, 10, 3))  # Placeholder for the image (replace with your image)
#
#     data_points = [
#         (np.array([1, 2.5, 3.4, 4, 5.8]), np.array([2, 3, 4, 5, 6])),
#         (np.array([1, 2, 3, 4, 5]), np.array([3, 4, 5, 6, 7])),
#         (np.array([2, 2, 2, 2, 2]), np.array([3, 4, 5, 6, 7])),
#         (np.array([1, 2, 3, 4, 5]), np.array([5, 6, 7, 8, 9])),
#     ]
#
#     # Filter out sets with NaN values
#     data_points = [(x, y) for x, y in data_points if not (np.any(np.isnan(x)) or np.any(np.isnan(y)))]
#     colors = ['red', 'blue', 'green', 'purple']
#
#     plot_data_and_line(image, data_points, colors)

# Function to fit a straight line using least squares
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# Generate some random data
data = np.random.rand(1000, 2)

# Perform K-means clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

# Get the cluster center and the distance of each data point to the nearest center
cluster_centers = kmeans.cluster_centers_
distances = pairwise_distances_argmin_min(data, cluster_centers)[1]

# Set a threshold to determine which cluster it belongs to based on the distance
threshold = 0.1
labels = np.where(distances < threshold, pairwise_distances_argmin_min(
    data, cluster_centers)[0], -1)

# Print the number of data points in each cluster
for i in range(kmeans.n_clusters):
    cluster_size = np.sum(labels == i)
    print(f"Cluster {i + 1} size: {cluster_size}")

# Here, you can process pixels belonging to the corresponding 
# semantic class based on the threshold and cluster label

# Generate some random data
data = np.random.rand(100, 2)

# Calculate the intra-cluster sum of squares for different K values
inertias = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

# Plotting the Elbow Rule
plt.plot(k_values, inertias, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()
