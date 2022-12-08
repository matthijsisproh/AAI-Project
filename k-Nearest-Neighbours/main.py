import numpy as np

from data import Data, data_min_max
from analyse import calculate_winrate
from knn import k_nearest_neighbours




if __name__ == "__main__":
    # Initialize the three datasets
    training_data = Data("dataset1.csv")
    validation_data = Data("validation1.csv")
    unclassified_data = Data("days.csv")

    # Assign labels to train and validation set
    training_data.assign_labels()
    validation_data.assign_labels()

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
    
    
    print(f"Calculating the k value with the best winrate from 1 to {len(training_data.data)}...")
    winrate, best_k = calculate_winrate(training_data, validation_data, len(training_data.data))
    print(f"The best winrate of {winrate}% is gained with a k of {best_k}.")
    print(f"This gives the KNN an error rate of {100-winrate}%.")
    
    results = []
    for point in unclassified_data.data:
        results.append(k_nearest_neighbours(data=training_data.data, labels=training_data.labels, point=point, k=best_k))
    print(f"The K Nearest Neighbours algorithm guesses thes to be the labels for the 10 unknown days: \n{results}")


