from data import Data, data_min_max

import numpy as np

###TODO###
def create_random_centroids(data, k):
    return 0

###TODO###
def validate_clusters(clustered_data, centroids, amount_data):
    return 0

###TODO###
def best_clustering(data, k, frequency):
    return 0

###TODO###
def KMeans(data, centroids):
    return 0

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




    



