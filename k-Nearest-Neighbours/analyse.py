from knn import k_nearest_neighbours

class analyse:
    def __init__(self, training_data, validation_data, range_k):
        """
        This function is used to find the best k value for the k-nearest neighbors algorithm.
        It will iterate through the range of k values and find the k value that has the highest
        winrate on the validation data.

        :param training_data: The training data that will be used to train the k-nearest neighbors algorithm.
        :param validation_data: The validation data that will be used to test the k-nearest neighbors algorithm.
        :param range_k: The range of k values that will be tested.
        :return: The best k value.
        """
        self.training_data = training_data
        self.validation_data = validation_data

        self.results = []
        self.best_k = 0
        self.k_winrates = {}
        self.range_k = range_k
        

    def print_progress(self, k):
        """
        This function prints a progress bar to the console.

        :param k: The current iteration of the loop.
        :return: None
        """
        percentage = round(k/(self.range_k-1)*100, 2)
        print('Progress: [{}{}{}]  {}%'.format( 
                ('=' * int(percentage//10) ), 
                ('>' if percentage < 100 else ''), 
                ('.' * int(10-(((percentage)//10))-1)), 
                percentage 
                ), end='\r'
                )

    def get_result(self, k):
        """
        This function takes in the training data, validation data, and k value.
        It then calculates the accuracy of the k-nearest neighbours algorithm
        using the validation data.

        :param k: The number of neighbours to use.
        :return results: The accuracy of the algorithm.
        """
        results = []
        for data_index, point in enumerate(self.validation_data.data):
                result = k_nearest_neighbours(self.training_data.data, self.training_data.labels, point, k)
                if result == self.validation_data.labels[data_index]:
                    results.append(True)
                else:
                    results.append(False)
        
        return results


    def calculate_winrate(self, k, results):
        """
        Calculates the winrate for a given k value.
    
        Parameters
        ----------
        k : int
            The k value to calculate the winrate for.
        results : list
            A list of the results of the kNN algorithm for the given k value.
            
        Returns
        -------
        None
    
        """

        self.k_winrates[k] = (sum(results) / len(self.validation_data.data)) * 100


    def winner(self):
        """
        This function returns the best k value for the KNN algorithm.
    
        Parameters:
        self (object): The object that is calling the function.
        
        Returns:
        self.best_k (int): The best k value for the KNN algorithm.
        """

        self.best_k = max(self.k_winrates, key=self.k_winrates.get)
        

    def run(self):
        """
        This function runs the simulation for the given range of k.
        It prints the progress of the simulation and calculates the winrate for each k.
        It also prints the winner of the simulation.
        
        Parameters
        ----------
        self : object
            The object of the class.
        range_k : int
            The range of k.
        
        Returns
        -------
        None
        """
        for k in range(1, self.range_k):
            self.print_progress(k)
            results = self.get_result(k)
            self.calculate_winrate(k, results)
        self.winner()


    def print_result(self):
        """
        Prints the best winrate and the k value that achieved it.
    
        Parameters
        ----------
        self : object
            The object that contains the best_k and k_winrates attributes.
        
        Returns
        -------
        None
        """
        print(f"The best winrate of {self.k_winrates[self.best_k]}% is gained with a k of {self.best_k}.")
