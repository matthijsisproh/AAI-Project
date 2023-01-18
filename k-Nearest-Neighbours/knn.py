from collections import Counter


def sort_dict(dict):
    """
    This function takes a dictionary as an argument and returns a dictionary
    with the same keys and values, but sorted by key.
    
    :param dict: The dictionary to be sorted.
    :return ordered_dict: The sorted dictionary.
    """
    ordered_dict = {}
    for key in sorted(dict):
        ordered_dict[key] = dict[key]

    return ordered_dict


def k_nearest_neighbours(data, labels, point, k):
    """
    This function takes in a dataset, the labels for the dataset, a point to be classified, and the number of neighbours to be considered.
    It then calculates the Euclidean distance between the point and each datapoint in the dataset, and returns the label of the closest k neighbours.
    If there is a tie, it will recursively call itself with k-1 neighbours until there is no tie.
    
    :param data: A list of lists, each containing the weather data
    :param labels: A list of labels for each datapoint in the dataset.
    :param point: A datapoint to be classified.
    :param k: The number of neighbours to be considered.
    :return str: The label of the closest k neighbours.
    """
    distances = {}
    for data_index, datapoints in enumerate(data):
        total_point_distance = 0 
        for datapoint_index, datapoint in enumerate(datapoints):
            total_point_distance += (datapoint - point[datapoint_index])**2    # Calculate Euclidean Distance
        distances[total_point_distance] = labels[data_index]

    ordered_distances = sort_dict(distances)
    closest = list(ordered_distances.values())[0:k]
    occurrences = Counter(closest)
    most_likely = occurrences.most_common()[0]

    if len(occurrences.most_common()) == 1 or most_likely[1] != occurrences.most_common()[1][1]:
        return most_likely[0]
    else:
        return k_nearest_neighbours(data, labels, point, k-1)

