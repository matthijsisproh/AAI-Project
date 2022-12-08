from collections import Counter


def sort_dict(dict):
    ordered_dict = {}
    for key in sorted(dict):
        ordered_dict[key] = dict[key]
    return ordered_dict


def k_nearest_neighbours(data, labels, point, k):
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

