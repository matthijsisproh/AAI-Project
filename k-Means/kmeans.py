import random
import copy
import numpy as np


def create_random_centroids(data, k):
    """
    This function creates k random centroids from the data.

    :param data: The data to create centroids from.
    :param k: The number of centroids to create.
    :return centroids: A list of k centroids.
    """
    centroids = []
    used_index = []
    for not_used in range(0, k):
        centroid_index = random.randrange(len(data))
        while centroid_index in used_index:
            centroid_index = random.randrange(len(data))
        
        centroids.append(data[centroid_index])
        used_index.append(centroid_index)
        
    return centroids


def validate_clusters(clustered_data, centroids, amount_data):
    """
    This function validates the clusters by calculating the total distance between the centroids and the data points.
    The total distance is divided by the amount of data points to get the average distance.

    :param clustered_data: A list of lists containing the data points in each cluster.
    :param centroids: A list of the centroids of the clusters.
    :param amount_data: The amount of data points.
    :return: The average distance between the centroids and the data points.
    """
    total_distance = 0
    for cluster_index, centroid in enumerate(centroids):
        for cluster in clustered_data[cluster_index]:
            total_distance += sum((centroid-cluster)**2)

    return total_distance/amount_data


def best_clustering(data, k, frequency):
    """
    This function takes in a dataset, a number of clusters, and a frequency.
    It then runs the KMeans algorithm on the dataset with the given number of clusters,
    and returns the best error rate out of the frequency number of runs.
    The best error rate is the lowest error rate.

    :param data: A list of lists, each sublist containing two floats.
    :param k: The number of clusters to use.
    :param frequency: The number of times to run the KMeans algorithm.
    :return best_error_rate: The best error rate out of the frequency number of runs.
    """
    best_error_rate = float("inf")
    for not_used in range(frequency):
        centroids = create_random_centroids(data, k)
        clustered_data, centroids = KMeans(data, centroids)
        error_rate = validate_clusters(clustered_data, centroids, amount_data=len(data))
        if error_rate < best_error_rate:
            best_error_rate = error_rate
    return best_error_rate


def KMeans(data, centroids):
    """
    This function takes in a list of data points and a list of centroids.
    It then clusters the data points into k clusters, where k is the number of centroids.
    It returns a list of clusters, where each cluster is a list of data points.
    It also returns the final centroids.

    :param data: A list of data points.
    :param centroids: A list of centroids.
    :return clustered_data: A list of clusters, where each cluster is a list of data points.
    :return centroids: The final centroids.
    """
    clustered_data = []
    k = len(centroids)
    temp_centroids = copy.deepcopy(centroids)

    for not_used in range(k):
        clustered_data.append([])

    for datapoint in data:
        nearest_centroid = (float("inf"), 0)
        for centroid_index, centroid in enumerate(centroids):
            distance = 0
            for data_iterator in range(len(data[0])):
                distance += (datapoint[data_iterator] - centroid[data_iterator])**2
            if distance < nearest_centroid[0]:
                nearest_centroid = (distance, centroid_index)
        clustered_data[nearest_centroid[1]].append(datapoint)
    
    for cluster_iterator, cluster in enumerate(clustered_data):
        if len(cluster)==0:
            continue
        mean = [0]*len(cluster[0])
        for datapoints in cluster:
            for data_iterator, datapoint in enumerate(datapoints):
                mean[data_iterator] += (datapoint/len(cluster))
        centroids[cluster_iterator] = mean

    if np.array_equal(temp_centroids, centroids):
        return clustered_data, centroids
    else: # Try cluster again
        return KMeans(data, centroids)
