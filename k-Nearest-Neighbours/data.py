import numpy as np

class Data:
    def __init__(self, filename):
        """
        This function takes a filename as input, and returns a dataframe. 
        The dataframe contains-- 'FG','TG','TN','TX','SQ', 'DR', 'RH' which
        represents the weather data of a day. The dataframe is indexed by date.

        -FG: day average windspeed (in 0.1 m/s).
        -TG: day average temperature (in 0.1 degrees Celsius).
        -TN: minimum temperature of day (in 0.1 degrees Celsius).
        -TX: maximum temperature of day (in 0.1 degrees Celsius).
        -SQ: amount of sunshine that day (in 0.1 hours). -1 for less than 0.05 hours.
        -DR: total time of precipitation (in 0.1 hours).
        -RH: total sum of precipitation that day (in 0.1 mm). -1 for less than 0.05mm.

        :param filename: A string representing the filename of the dataset.
        """
        self.data = np.genfromtxt(
            f'C:\\Users\\MatthijsKoelewijnDen\\Documents\\Github\\AAI-Project\\data\\{filename}', 
            delimiter=';', 
            usecols=[1,2,3,4,5,6,7], 
            converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)}
            )

        self.dates = np.genfromtxt(
            f'C:\\Users\\MatthijsKoelewijnDen\\Documents\\Github\\AAI-Project\\data\\{filename}', 
            delimiter=';', 
            usecols=[0]
            )
        
        self.labels = []

    def assign_labels(self):
        """
        This function assigns labels to the dates in the dataset.
        The labels are:
            - winter: from 01-01 to 03-01
            - lente: from 03-01 to 06-01
            - zomer: from 06-01 to 09-01
            - herfst: from 09-01 to 12-01
            - winter: from 12-01 to end of year
            
        :param self: The object that contains the dates and labels
        :return None
        """
        year = int(str(self.dates[0])[0:4])*10000 
        for label in self.dates:
            if label < (year + 301):
                self.labels.append('winter')
            elif (year + 301) <= label < (year + 601):
                self.labels.append('lente')
            elif (year + 601) <= label < (year + 901):
                self.labels.append('zomer')
            elif (year + 901) <= label < (year + 1201):
                self.labels.append('herfst')
            else: # from 01-12 to end of year
                self.labels.append('winter')


    def normalise(self, minimum_list, maximum_list):
        """
        This function normalises the data in the data set.
        It takes the minimum and maximum values of each data point and uses them to normalise the data.
        The amount of sunshine in a day is made 1.45 times as important as the other data.
        This gave us the best results.

        :param self: The object that contains the dates and labels
        :param minimum_list: A list object with all minimum values of all datasets for normalisation
        :param maximum_list: A list object with all maximum values of all datasets for normalisation
        :return None
        """
        for data_index, datapoints in enumerate(self.data):
            for datapoint_index, datapoint in enumerate(datapoints):
                if datapoint_index == 4:
                    self.data[data_index][datapoint_index] = ((datapoint - minimum_list[datapoint_index]) / (maximum_list[datapoint_index] - minimum_list[datapoint_index]))*1.45
                else:
                    self.data[data_index][datapoint_index] = (datapoint - minimum_list[datapoint_index]) / (maximum_list[datapoint_index] - minimum_list[datapoint_index])


def data_min_max(data):
    """
    This function takes a list of lists as input and returns a list of minimum values and a list of maximum values.
    
    :param data: A list of lists, each containing the weather data
    :return minimum_list: A list of minimum values of all given data.
    :return maximum_list: A list of maximum values of all given data.
    """
    minimum_list = [float('inf')] * 7
    maximum_list = [float('-inf')] * 7
    for datapoints in data:
        for data_index, datapoint in enumerate(datapoints):
            if datapoint < minimum_list[data_index]:
                minimum_list[data_index] = datapoint
            elif datapoint > maximum_list[data_index]:
                maximum_list[data_index] = datapoint
    return minimum_list, maximum_list