import random
import copy
import numpy as np


def create_random_centroids(data, k):
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
    total_distance = 0
    for cluster_index, centroid in enumerate(centroids):
        for cluster in clustered_data[cluster_index]:
            total_distance += sum((centroid-cluster)**2)

    return total_distance/amount_data


def best_clustering(data, k, frequency):
    best_error_rate = float("inf")
    for not_used in range(frequency):
        centroids = create_random_centroids(data, k)
        clustered_data, centroids = KMeans(data, centroids)
        error_rate = validate_clusters(clustered_data, centroids, amount_data=len(data))
        if error_rate < best_error_rate:
            best_error_rate = error_rate
    return best_error_rate


def KMeans(data, centroids):
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
