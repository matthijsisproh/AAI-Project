from neuralnetwork import NeuralNetwork
import matplotlib.pyplot as plt
import os

ZETA = 0.1

def read_data(validation_size):
    """
    This functions reads the data from the file and splits it into training and validation data.

    :param validation_size: The number of data points to be used for validation
    :return all_data: A list of all the data points.
    :return train_data: A list of the training data points.
    :return validation_data: A list of the validation data points.
    
    """

    split_index = [0, 50, 100] # where to split the data based on the labels
    train_data = []
    validation_data = []
    with open (f"{os.getcwd()}/ProgrammingNNs/data/iris.data", "r") as myfile:
        data = myfile.read().splitlines()
    all_data = []
    for i, datapoint in enumerate(data):
        all_data.append(datapoint.split(","))
        all_data[i] = [float(all_data[i][0]), float(all_data[i][1]), float(all_data[i][2]), float(all_data[i][3]), all_data[i][4]]
    for i in range(len(all_data)):
        if (i < split_index[0] + validation_size and i >= split_index[0]) or (i < split_index[1] + validation_size and i >= split_index[1]) or (i < split_index[2] + validation_size and i >= split_index[2]):
            validation_data.append(all_data[i])
        else:
            train_data.append(all_data[i])
    return all_data, train_data, validation_data
    

def data_min_max(data):
    """
    This function takes a list of lists as input and returns a list of minimum values and a list of maximum values.
    
    :param data: A list of lists, each containing the weather data
    :return minimum_list: A list of minimum values of all given data.
    :return maximum_list: A list of maximum values of all given data.
    """
    minimum_list = [float('inf')] * 4
    maximum_list = [float('-inf')] * 4
    for datapoints in data:
        for index in range(0, len(datapoints)-1):
            if datapoints[index] < minimum_list[index]:
                minimum_list[index] = datapoints[index]
            elif datapoints[index] > maximum_list[index]:
                maximum_list[index] = datapoints[index]
    return minimum_list, maximum_list


def normalise(data, minimum_list, maximum_list):
    """
        This function normalises the data in the data set.

        :param data: A list object with all data points
        :param minimum_list: A list object with all minimum values of all datasets for normalisation
        :param maximum_list: A list object with all maximum values of all datasets for normalisation
        :return data: A list object with all normalised data points 
    """
    for index_data, datapoints in enumerate(data):
        for index in range(len(datapoints)-1):
            data[index_data][index] = (datapoints[index] - minimum_list[index]) / (maximum_list[index] - minimum_list[index])
    return data


def get_desired_output(input):
    """
    This function takes in a list of 5 elements, the first 4 being the features of an iris flower, and the last being the
    class of the flower. It returns a list of 3 elements, each element being a binary value. The first element is 1 if the
    flower is of the class Iris-setosa, 0 otherwise. The second element is 1 if the flower is of the class Iris-versicolor,
    0 otherwise. The third element is 1 if the flower is of the class Iris-virginica, 0 otherwise.

    :param input: A datapoint
    :return: desired output of datapoint
    """
    if input[-1] == "Iris-setosa":
        return [1, 0, 0]
    elif input[-1] == "Iris-versicolor":
        return [0, 1, 0]
    elif input[-1] == "Iris-virginica":
        return [0, 0, 1]


def validation(network, validation_data):
    """
    This function takes a network and a validation data set as input.
    It then calculates the succes rate of the network on the validation data set.
    It returns the success rate and the number of correct answers.

    :param network: The network to be tested.
    :param validation_data: The validation dataset.
    :return success_rate: The success rate of the network on the validation data set.
    :return correct_answers: The number of correct answers
    """
    sum_of_neuron = 0
    correct_answers = 0
    for datapoint in validation_data:
        network.set_input_layer(datapoint[0:-1])

        desired_output = get_desired_output(datapoint)
        actual_output = network.get_output()
        y__a =  [y_i - a_i for y_i, a_i in zip(desired_output, actual_output)] # subtract arrays
        squared = [x**2 for x in y__a] # square the array
        sum_of_neuron += sum(squared)

        if actual_output.index(max(actual_output)) == desired_output.index(1):
            correct_answers += 1
    
    success_rate = correct_answers/len(validation_data)*100 
    return success_rate, correct_answers


NN_data, training_data, validation_data = read_data(validation_size=17)


minimum_list, maximum_list = data_min_max(NN_data)

training_data = normalise(training_data, minimum_list, maximum_list)
validation_data = normalise(validation_data, minimum_list, maximum_list)


network = NeuralNetwork()
network.set_input_layer(training_data[0][0:-1])
network.add_layer(4)
network.add_layer(3)

# Train amount of epochs
success_rate_list = []
n_epoch= 1000
for epoch in range(n_epoch):
    # just a progress bar
    percentage = round(epoch/(n_epoch-1)*100, 2)
    print('Progress: [{}{}{}]  {}%'.format( ('=' * int(percentage//10) ), ('>' if percentage < 100 else ''), ('.' * int(10-(((percentage)//10))-1)), percentage ), end='\r')
    for training_data_index in range(0, len(training_data)):
        results = []
        network.set_input_layer(training_data[training_data_index][0:-1])
        # network.get_output()
        network.forward_propagation(get_desired_output(training_data[training_data_index]))
        network.backwards_propagation()
    ZETA *= 0.9995
    network.update_zeta(ZETA)
    success_rates, correct_guesses = validation(network, validation_data)
    success_rate_list.append(success_rates)

# Plot success rate
plt.plot(range(len(success_rate_list)), success_rate_list, linewidth=1, color='blue', label='Success Rate per Epoch')
plt.legend()
plt.show()

# Success rate final epoch
success_rates, correct_guesses = validation(network, training_data)
print(f"The final successrate is {round(success_rates, 2)}%                       ")
print(f"The network has correctly guessed {correct_guesses} out of {len(training_data)} datapoints")

# Output
# `
# The final successrate is 94.95%
# The network has correctly guessed 94 out of 99 datapoints
# `
