import numpy as np
from sklearn.cluster import KMeans
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time
import os


def post_process_clusters(labels, min_cluster_size):
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    # Mark clusters smaller than the minimum cluster size as noise
    noise_labels = unique_labels[label_counts < min_cluster_size]

    # Mark noise points as zero
    for noise_label in noise_labels:
        labels[labels == noise_label] = 0

    return labels


def lane_detection(binary_image_path, instance_image_path, num_clusters, min_cluster_size=100):
    # Read binary image and feature vector image
    binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)

    # Set a threshold value, set pixels greater than the threshold value to white,
    # and pixels less than or equal to the threshold value to black
    _, binary_smoothed = cv2.threshold(
        binary_image, 200, 255, cv2.THRESH_BINARY)

    # Use median filtering for smoothing
    binary_smoothed = cv2.medianBlur(binary_smoothed, 5)
    instance_image = cv2.imread(instance_image_path)

    # test(Use binary_image directly for clustering)
    # binary_test = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)

    # Get the coordinates of non-zero pixels
    non_zero_points = np.column_stack(np.where(binary_smoothed > 0))

    # Get feature vector
    features = [instance_image[point[0], point[1]]
                for point in non_zero_points]
    # features = binary_test

    # Convert to numpy array
    features = np.array(features)

    # Using K-means clustering
    clustering = KMeans(n_clusters=num_clusters, random_state=42)
    labels = clustering.fit_predict(features)

    # Create a crop row assignment dictionary
    lanes = {label: [] for label in set(labels)}

    # Post-processing: Removing small clusters
    labels = post_process_clusters(labels, min_cluster_size)

    # Example result image after creating clustering
    instance_result = np.zeros_like(binary_image)

    # Pixel labeling based on crop row assignment results
    for i, point in enumerate(non_zero_points):
        # for i, point in enumerate(binary_test):
        label = labels[i]
        lanes[label].append(point)
        instance_result[point[0], point[1]] = label + 1

    # Save the instance result image
    # cv2.imwrite("instance_result.jpg", instance_result)
    opening = cv2.morphologyEx(
        instance_result, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8))

    # Display example result image
    # plt.imshow(opening, cmap='jet')  # Use jet colormap to show different instances
    # plt.title('Instance Segmentation Result')
    # plt.colorbar()
    # plt.show()
    labels = np.unique(labels, axis=0)
    return instance_result, labels
    # return opening, labels


def extract_centers(binary_image, labels, num_bars=9):
    # Divide into strips
    bar_height = binary_image.shape[0] // num_bars

    centers_1 = []
    centers_2 = []
    centers_3 = []
    centers_4 = []

    centers_mid = []   # Temporarily save intermediate lists

    for i in range(num_bars):
        # Extract each strip
        bar = binary_image[i * bar_height: (i + 1) * bar_height, :]

        for l in labels:
            # Extract center point
            D = np.sum(bar == (l + 1))
            if D == 0:
                continue
            center_x = np.sum(np.where(bar == (l + 1))
                              [1]) / np.sum(bar == (l + 1))
            if np.isnan(center_x):
                continue
            center_y = i * bar_height + bar_height // 2
            if l + 1 == 1:
                centers_1.append((center_x, center_y))
            elif l + 1 == 2:
                centers_2.append((center_x, center_y))
            elif l + 1 == 3:
                centers_3.append((center_x, center_y))
            else:
                centers_4.append((center_x, center_y))
    centers_mid = adjust_centers_order(
        centers_1, centers_2, centers_3, centers_4)
    # return centers_1, centers_2, centers_3, centers_4
    return centers_mid


# Arrange the detected crop rows in the order of categories 1, 2, 3, 4
def adjust_centers_order(centers_1, centers_2, centers_3, centers_4):
    # Find the group of elements with the largest center_y value in each array
    max_centers = [
        max(centers_1, key=lambda x: x[1]),
        max(centers_2, key=lambda x: x[1]),
        max(centers_3, key=lambda x: x[1]),
        max(centers_4, key=lambda x: x[1])
    ]

    # Sort by the size of center_x
    sorted_max_centers = sorted(max_centers, key=lambda x: x[0])

    # Adjust the order of the four arrays centers_1, centers_2, centers_3, and centers_4,
    # and temporarily save the adjusted order to the centers_mid array
    centers_mid = []
    for center in sorted_max_centers:
        if center in centers_1:
            centers_mid.append(centers_1)
        elif center in centers_2:
            centers_mid.append(centers_2)
        elif center in centers_3:
            centers_mid.append(centers_3)
        elif center in centers_4:
            centers_mid.append(centers_4)

    # Returns the adjusted four arrays and the temporarily saved adjusted order array
    return centers_mid


# Function to fit a straight line using least squares
def fit_line(x, y):
    if np.all(x == x[0]):  # Check if x-values are constant
        m = np.inf  # Slope is set to infinity for constant x-values
        c = np.mean(y)  # Intercept is the mean of y-values
    else:
        A = np.vstack([x, np.ones_like(x)]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c


# Function to find extended points for a line that starts from one edge and ends at the opposite edge
def find_extended_points(m, c, image_shape):
    x1 = 0
    y1 = c
    x2 = image_shape[1]
    y2 = m * x2 + c

    return x1, y1, x2, y2


# Function to plot data points and extended lines on an image
def plot_data_and_line(image, data_points, line_width=1.0, idx=0):
    plt.figure()  # Create a new image object
    # whiteboard = np.ones_like(image) * 255
    # plt.imshow(whiteboard)
    plt.imshow(image)  # Display the image

    for i, (x, y) in enumerate(data_points):
        m, c = fit_line(x, y)
        if np.isinf(m):
            # plt.axvline(x[0], color=colors[i], label=f'Line {i + 1}', linewidth=line_width)  # with label
            plt.axvline(x[0], color=colors[i], linewidth=line_width)
        else:
            # plt.plot(x, m * x + c, color=colors[i], label=f'Line {i + 1}')
            # plt.scatter(x, y, color=colors[i])  # Scatter plot for data points

            # Find extended points for the line
            x1, y1, x2, y2 = find_extended_points(m, c, image.shape)

            # Plot the original line
            # plt.plot(x, m * x + c, color=colors[i], label=f'Line {i + 1}', linewidth=line_width)  # with label
            plt.plot(x, m * x + c, color=colors[i], linewidth=line_width)
            # plt.scatter(x, y, color=colors[i], facecolors='none')  # Scatter plot for data points

            # Plot the extended line
            # plt.plot([x1, x2], [y1, y2], linestyle='--', color=colors[i], linewidth=line_width)
            plt.plot([x1, x2], [y1, y2], color=colors[i], linewidth=line_width)

    # Set axis limits for lower left corner as origin
    plt.xlim((0, image.shape[1]))
    plt.ylim((image.shape[0], 0))

    # Set the axis to be invisible
    plt.axis('off')

    # plt.legend()
    plt.savefig('E:/A_trans/results/AH/3/Unet/with_img/' +
                f'{idx}.png', format='png', dpi=400, bbox_inches='tight')
    # plt.show()
    plt.close()  # Close the current image object to free up memory


# Split this function into draw_line
def fit_lines_hough(data_points_old, idx):
    # Convert to the new data_points format
    data_points_new = [(np.array([x for x, y in points]), np.array([y for x, y in points])) for points in
                       data_points_old]

    # Filter out sets with NaN values
    # Remove individual data points with NaN values
    # data_points_new = [
    #     (
    #         np.array([xi for xi, yi in zip(x, y) if not np.isnan(xi) and not np.isnan(yi)]),
    #         np.array([yi for xi, yi in zip(x, y) if not np.isnan(xi) and not np.isnan(yi)])
    #     )
    #     for x, y in data_points_new
    # ]
    # # Remove sets with empty arrays resulting from the removal of NaN values
    # data_points_new = [(x, y) for x, y in data_points_new if len(x) > 0 and len(y) > 0]

    image = IMG.copy()
    plot_data_and_line(image, data_points_new, line_width=2.0, idx=idx)


def draw_lines(instance_result, labels, idx=0):
    start_time = time.time()  # Recording start time

    # Extract center point
    centers = extract_centers(instance_result, labels)
    # idx = 0

    # Perform Hough transform to fit the straight line
    fit_lines_hough(centers, idx)
    end_time = time.time()  # Record end time
    # The time taken to calculate (milliseconds)
    elapsed_time_ms = (end_time - start_time) * 1000
    print("Time consumed (milliseconds) line fitting:", elapsed_time_ms)

    # Visualize the results
    # cv2.polylines(IMG, [np.int32(lines)], isClosed=False, color=(255, 0, 0), thickness=2)

    # Show results
    # cv2.imshow("Result", RGB_IMG)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


colors_map = [(33, 150, 243), (255, 152, 0), (244, 67, 54),
              (0, 150, 136), (63, 81, 181), (76, 175, 80)]
colors = ['#2196F3', '#FF9800', '#F44336', '#009688', '#3F51B5', '#4CAF50']
num_clusters = 4  # Number of clusters

# Loop call
img_id = 106  # Start ID
file_path = 'E:/A_trans/results/AH/3/Unet/'
for index in range(100):
    if os.path.exists(file_path + f'{img_id}' + '_input.png'):
        binary_image_path = file_path + f'{img_id}' + '_binary_output.png'
        instance_image_path = file_path + f'{img_id}' + '_instance_output.png'
        RGB_IMG = cv2.imread(file_path + f'{img_id}' + '_input.png')
        IMG = cv2.cvtColor(RGB_IMG, cv2.COLOR_RGB2BGR)

        instance_out, labels = lane_detection(
            binary_image_path, instance_image_path, num_clusters)
        draw_lines(instance_out, labels, img_id)
        img_id += 2
    else:
        while True:
            # Construct image file path
            image_file = file_path + f'{img_id}' + '_input.png'

            # Determine whether a file exists
            if os.path.exists(image_file):
                print(f"Image File {image_file} exists, end the loop")
                break
            else:
                print(f"Image file {image_file} does not exist, continue the loop")
                img_id += 1
        binary_image_path = file_path + f'{img_id}' + '_binary_output.png'
        instance_image_path = file_path + f'{img_id}' + '_instance_output.png'
        RGB_IMG = cv2.imread(file_path + f'{img_id}' + '_input.png')
        IMG = cv2.cvtColor(RGB_IMG, cv2.COLOR_RGB2BGR)

        instance_out, labels = lane_detection(
            binary_image_path, instance_image_path, num_clusters)
        draw_lines(instance_out, labels, img_id)
        img_id += 2
