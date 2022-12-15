from data import Data, data_min_max
from kmeans import best_clustering

import numpy as np
import matplotlib.pyplot as plt 


def plot_results(length, best):
    plt.plot(
        length, 
        best, 
        'o-', 
        linewidth=2, 
        color='red', 
        marker='o',
        markeredgecolor='#1e78b4',
        markerfacecolor='#1e78b4',
        markersize=5)

    plt.title("K means error rate")
    plt.xlabel("K")
    plt.ylabel("Error rate")
    plt.show()


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

    # Get the result
    num_iterations = 100 # Define number of iterations
    length = [k+1 for k in range(num_iterations)] # Fill list with k
    best_clusters = [best_clustering(training_data.data, k+1, frequency=10) for k in range(num_iterations)] # Call best_clustering function and append to list
    plot_results(length, best_clusters)




