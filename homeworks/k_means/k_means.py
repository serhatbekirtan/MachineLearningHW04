from typing import Tuple

import numpy as np
import time

from numpy import ndarray

from utils import problem, load_dataset


@problem.tag("hw4-A") # Done
def calculate_centers(data: np.ndarray, classifications: np.ndarray, num_centers: int) -> np.ndarray:
    """
    Sub-routine of Lloyd's algorithm that calculates the centers given datapoints and their respective classifications/assignments.
    num_centers is additionally provided for speed-up purposes.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        classifications (np.ndarray): Array of shape (n,) full of integers in range {0, 1, ...,  num_centers - 1}.
            Data point at index i is assigned to classifications[i].
        num_centers (int): Number of centers for reference.
            Might be usefull for pre-allocating numpy array (Faster that appending to list).

    Returns:
        np.ndarray: Array of shape (num_centers, d) containing new centers.
    """

    new_centroids = np.zeros((num_centers, data.shape[1]))
    # Select the data points that are assigned to class j.
    for j in range(0, num_centers):
        # compute centroids
        J = np.where(classifications == j)
        data_C = data[J]
        # Calculate the mean of the points and set it as the center.
        new_centroids[j] = data_C.mean(axis=0)

    return new_centroids


@problem.tag("hw4-A") # Done
def cluster_data(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Sub-routine of Lloyd's algorithm that clusters datapoints to centers given datapoints and centers.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        centers (np.ndarray): Array of shape (k, d). Each row is a center to which a datapoint can be clustered.

    Returns:
        np.ndarray: Array of integers of shape (n,), with each entry being in range {0, 1, 2, ..., k - 1}.
            Entry j at index i should mean that j^th center is the closest to data[i] datapoint.
    """

    n = data.shape[0]
    k = centers.shape[0]

    # Initialize the distances array.
    distances = np.zeros((n, k), dtype=float)

    # Compute the distances between each point and each center. And store it in the distances array.
    for d, data_v in enumerate(data):
        for centroid, centroid_v in enumerate(centers):
            distance = np.linalg.norm(data_v - centroid_v)
            distances[d, centroid] = distance

    # Determine nearest cluster
    nearest_center = np.argmin(distances, axis=1)

    return nearest_center

@problem.tag("hw4-A") # Done
def calculate_error(data: np.ndarray, centers: np.ndarray) -> float:
    """Calculates error/objective function on a provided dataset, with trained centers.

    Args:
        data (np.ndarray): Array of shape (n, d). Dataset to evaluate centers on.
        centers (np.ndarray): Array of shape (k, d). Each row is a center to which a datapoint can be clustered.
            These should be trained on training dataset.

    Returns:
        float: Single value representing mean objective function of centers on a provided dataset.
    """

    n = data.shape[0]
    k = centers.shape[0]

    nearest_center = cluster_data(data, centers)

    se = 0
    for data_v, center_idx in zip(data, nearest_center):
        center = centers[int(center_idx)]
        distance = np.linalg.norm(data_v - center)
        se += distance

    error = se / n

    return error

@problem.tag("hw4-A") # Done
def lloyd_algorithm(data: np.ndarray, num_centers: int, epsilon: float = 10e-3) -> Tuple[ndarray, ndarray, float]:
    """Main part of Lloyd's Algorithm.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        num_centers (int): Number of centers to train/cluster around.
        epsilon (float, optional): Epsilon for stopping condition.
            Training should stop when max(abs(centers - previous_centers)) is smaller or equal to epsilon.
            Defaults to 10e-3.

    Returns:
        np.ndarray: Array of shape (num_centers, d) containing trained centers.

    Note:
        - For initializing centers please use the first `num_centers` data points.
    """

    I = np.random.choice(data.shape[0], num_centers)
    centroids = data[I, :]
    classifications = np.zeros(data.shape[0], dtype=np.int64)

    max_iter = 4
    loss = 0

    for m in range(0, max_iter):
        # Cluster the data points to the centers
        classifications = cluster_data(data, centroids)
        # Calculate the new centers
        new_centroids = calculate_centers(data, classifications, num_centers)
        # Calculate the loss
        new_loss = calculate_error(data, new_centroids)
        obj_func = np.abs(centroids - new_centroids)

        # Stopping criterion
        if np.max(obj_func) < epsilon:
            return new_centroids, classifications, new_loss

        centroids = new_centroids
        loss = new_loss

        print(loss)

    return centroids, classifications, loss