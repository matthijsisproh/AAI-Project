import math
from neighbour import Neighbour
from vector import Vector
from collections import Counter

### TODO EUCLIDEAN DISTANCE CODE
 
def compute_distance(point_x, point_y):
    """
    Function to return the euclidean distance of point x and point y
    """
    distance = math.dist(point_x, point_y)
    return distance


def most_frequent_label(neighbours):
    """
    Function to return the most frequent label of given list of neighbours
    """
    winning_label = ""
    k_labels = []

    for k_count in range(0, k):
        dist = list_of_distances[k_count]
        for neighbour in neighbours:
            if neighbour._distance == dist:
                k_labels.append(neighbour._label)

    most_occur = Counter(k_labels).most_common(4)
    return winning_label


def sort_list(list):
    """
    Function to return sorted list with descending values.
    """
    sorted_list = sorted(list)

    return sorted_list


def k_nearest_neighbours(training_data, test_point, k):
    """
    Function to return the nearest neighbour of given test point, 
    where the nearest neighbour is the nearest point computed by 
    euclidean distance.  
    """

    n_data = training_data._normalized_data
    labels = training_data.create_labels()
    list_of_distances = []
    list_of_vec = []

    for index in range(0, len(n_data)):        
        distance = compute_distance(n_data[index], test_point)

        vec = Vector(n_data[index], labels[index], distance)
        list_of_distances.append(distance)
        list_of_vec.append(vec)

    list_of_distances = sort_list(list_of_distances) # Sorts list in descending order

    k_labels = []

    for k_count in range(0, k):
        dist = list_of_distances[k_count]
        for vec in list_of_vec:
            if vec._distance == dist:
                k_labels.append(vec._label)

    most_occur = Counter(k_labels).most_common(4)

    is_tie = False
    max = 0
    winning_label = ""
    for index in range(0, len(most_occur)):
        if max == most_occur[index][1]:
            is_tie = True

        if max < most_occur[index][1]:
            max = most_occur[index][1]
            winning_label = most_occur[index][0]
        
    
    print(winning_label)
    if(is_tie):
        print(most_occur)



    

