from knn import k_nearest_neighbours

class analyse:
    def __init__(self, training_data, validation_data, range_k):
        self.training_data = training_data
        self.validation_data = validation_data

        self.results = []
        self.best_k = 0
        self.k_winrates = {}
        self.range_k = range_k
        

    def print_progress(self, k):
        percentage = round(k/(self.range_k-1)*100, 2)
        print('Progress: [{}{}{}]  {}%'.format( 
                ('=' * int(percentage//10) ), 
                ('>' if percentage < 100 else ''), 
                ('.' * int(10-(((percentage)//10))-1)), 
                percentage 
                ), end='\r'
                )

    def get_result(self, k):
        results = []
        for data_index, point in enumerate(self.validation_data.data):
                result = k_nearest_neighbours(self.training_data.data, self.training_data.labels, point, k)
                if result == self.validation_data.labels[data_index]:
                    results.append(True)
                else:
                    results.append(False)
        
        return results


    def calculate_winrate(self, k, results):
        self.k_winrates[k] = (sum(results) / len(self.validation_data.data)) * 100


    def winner(self):
        self.best_k = max(self.k_winrates, key=self.k_winrates.get)
        

    def run(self):
        for k in range(1, self.range_k):
            self.print_progress(k)
            results = self.get_result(k)
            self.calculate_winrate(k, results)
        self.winner()


    def print_result(self):
        print(f"The best winrate of {self.k_winrates[self.best_k]}% is gained with a k of {self.best_k}.")
