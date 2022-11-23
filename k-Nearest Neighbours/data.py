import numpy as np
from sklearn import preprocessing
from vector import Vector

class Data:
    def __init__(self, filename : str, year : str):
        self._path_to_file = "C:\\Users\\MatthijsKoelewijnDen\\Documents\\Github\\AAI-Project\\k-Nearest Neighbours\\data\\"
        self._data = np.genfromtxt(self._path_to_file + filename, 
                        delimiter=";", 
                        usecols=[1, 2, 3, 4, 5, 6, 7], 
                        converters={5: lambda s: 0 
                        if s == b"-1" 
                        else float(s), 7: lambda s: 0 if s == b"-1" 
                        else float(s)}
                        )       
        self._dates = np.genfromtxt(self._path_to_file + filename, delimiter=";", usecols=[0])
        self._labels = []
        self._year = year
        self._normalized_data = preprocessing.normalize(self._data)

    def create_labels(self):
        for label in self._dates:
            if label < int(self._year + "0301"):
                self._labels.append("winter")
            elif int(self._year + "0301") <= label < int(self._year + "0601"):
                self._labels.append("lente")
            elif int(self._year + "0601") <= label < int(self._year + "0901"):
                self._labels.append("zomer")
            elif int(self._year + "0901") <= label < int(self._year + "1201"):
                self._labels.append("herfst")
            else: # from 01-12 to end of year
                self._labels.append("winter")

        return self._labels
        
    def normalize_data(self):
        normalized_data = preprocessing.normalize(self._data)
        self._data = normalized_data
        # for index in range(0, len(self._data)):
        #     normalized_data = preprocessing.normalize([self._data[index]])
        #     print(normalized_data)


# test = Data("dataset1.csv", "2000")
# test.normalize_data()