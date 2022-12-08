import numpy as np

class Data:
    def __init__(self, filename):
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
        for data_index, datapoints in enumerate(self.data):
            for datapoint_index, datapoint in enumerate(datapoints):
                self.data[data_index][datapoint_index] = (datapoint - minimum_list[datapoint_index]) / (maximum_list[datapoint_index] - minimum_list[datapoint_index])
               

def data_min_max(data):
        minimum_list = [float('inf')] * 7
        maximum_list = [float('-inf')] * 7
        for datapoints in data:
            for data_index, datapoint in enumerate(datapoints):
                if datapoint < minimum_list[data_index]:
                    minimum_list[data_index] = datapoint
                elif datapoint > maximum_list[data_index]:
                    maximum_list[data_index] = datapoint
        return minimum_list, maximum_list