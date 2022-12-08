import numpy as np

from data import Data, data_min_max
from analyse import analyse
from knn import k_nearest_neighbours




if __name__ == "__main__":
    # Unclassified file
    filename_unclassified_data = "days.csv"

    # Initialize the three datasets
    training_data = Data("dataset1.csv")
    validation_data = Data("validation1.csv")
    unclassified_data = Data(filename_unclassified_data)

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
 

    # Analyse and calculate the winrate
    analyse_winrate = analyse(training_data, validation_data, range_k=len(training_data.data))
    analyse_winrate.run()
    analyse_winrate.print_result()
    

    # Apply the algorithm's best_k to guess the labels
    results = []
    for datapoint in unclassified_data.data:
        results.append(
            k_nearest_neighbours(
                data=training_data.data, 
                labels=training_data.labels, 
                point=datapoint, 
                k=analyse_winrate.best_k)
            )

    print(f"The KNN algorithm guesses these labels for file {filename_unclassified_data} : \n{results}" )
    