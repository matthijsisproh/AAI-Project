from data import Data, data_min_max
import random

import numpy as np
import math

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

###TODO###
def validate_clusters(clustered_data, centroids, amount_data):
    return 0

###TODO###
def best_clustering(data, k, frequency):
    return 0

###TODO###
def KMeans(data, centroids):
    cluster_list = []
    cluster = []
    for centroid_index in range(0, len(centroids)):
        min_dist = float('inf')
        for datapoint in data:
            distance = math.dist(datapoint, centroids[centroid_index])
            if min_dist > distance:
                min_dist = distance

            
        
        # find the nearest centroid
        # assign datapoint to cluster

    # for cluster in range (1, k):
    #     #calculate new centroid c[cluster] as the mean of all points datapoint that are assigned to cluster

        

###TODO###
def plot_results(length, best):
    return 0


if __name__ == "__main__":
    # Unclassified file
    filename_unclassified_data = "days.csv"

    # Initialize the three datasets
    training_data = Data("dataset1.csv")
    validation_data = Data("validation1.csv")
    unclassified_data = Data(filename_unclassified_data)

    # Find minimum and maximum value of all datasets for normalisation
    min_data, max_data = data_min_max(
        np.concatenate((            # Merge all datasets in to single array
            training_data.data,
            validation_data.data, 
            unclassified_data.data
            ))
        ) 

    # Normalise all datasets
    training_data.normalise(min_data, max_data)
    validation_data.normalise(min_data, max_data)
    unclassified_data.normalise(min_data, max_data)

    
    #TEST
    centroids = create_random_centroids(unclassified_data.data, 4)
    KMeans(training_data.data, centroids)

    
    




