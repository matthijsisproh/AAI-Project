import numpy as np
from typing import Union, List, Any
from data import Data
from knn import k_nearest_neighbours
from vector import Vector
import matplotlib.pyplot as plt

def plot_result(result):
    labels = ["lente", "zomer", "herfst", "winter"]
    values = [1, 10, 50, 10]
    plt.figure(figsize=(10, 4))

    plt.subplot(131)
    plt.bar(labels, values)
    plt.suptitle("Most common labels")
    plt.show()


if __name__ == "__main__":
    # Class
    training_NN= Data(filename="dataset1.csv", year="2000")
    validation_NN = Data(filename="validation1.csv", year="2001")

    test_class = Data("days.csv", "2020")
    test_set = test_class._normalized_data

    for test_point in test_set:
        k_nearest_neighbours(validation_NN, test_point, k=5)

