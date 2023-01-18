from neuralnetwork import NeuralNetwork
import matplotlib.pyplot as plt

ZETA = 0.1

def read_data(validation_size):
    split_index = [0, 50, 100] # where to split the data based on the labels
    train_data = []
    validation_data = []
    with open ("data/iris.data", "r") as myfile:
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
    

def find_min_max(data):
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
    for index_data, datapoints in enumerate(data):
        for index in range(len(datapoints)-1):
            data[index_data][index] = (datapoints[index] - minimum_list[index]) / (maximum_list[index] - maximum_list[index])
    return data


def get_desired_output(input):
    if input[-1] == "Iris-setosa":
        return [1, 0, 0]
    elif input[-1] == "Iris-versicolor":
        return [0, 1, 0]
    elif input[-1] == "Iris-virginica":
        return [0, 0, 1]


def validation(network, validation_data):
    sum_of_neuron = 0
    correctGuesses = 0
    for datapoint in validation_data:
        network.set_input_layer(datapoint[0:-1])

        desired_output = get_desired_output(datapoint)
        actual_output = network.get_output()
        y__a =  [y_i - a_i for y_i, a_i in zip(desired_output, actual_output)] # subtract arrays
        squared = [x**2 for x in y__a] # square the array
        sum_of_neuron += sum(squared)

        if actual_output.index(max(actual_output)) == desired_output.index(1):
            correctGuesses += 1
    
    successRate = correctGuesses/len(validation_data)*100 
    return successRate, correctGuesses


allData, trainingData, validationData = read_data(validation_size=17)


minimumList, maximumList = find_min_max(allData)

trainingData = normalise(trainingData, minimumList, maximumList)
validationData = normalise(validationData, minimumList, maximumList)


network = NeuralNetwork()
network.set_input_layer(trainingData[0][0:-1])
network.add_layer(4)
network.add_layer(3)

# training for given amount of epochs
successRates = []
numberOfEpochs = 1000
for epoch in range(numberOfEpochs):
    # just a progress bar
    percentage = round(epoch/(numberOfEpochs-1)*100, 2)
    print('Progress: [{}{}{}]  {}%'.format( ('=' * int(percentage//10) ), ('>' if percentage < 100 else ''), ('.' * int(10-(((percentage)//10))-1)), percentage ), end='\r')
    for trainingDataIndex in range(0, len(trainingData)):
        results = []
        network.set_input_layer(trainingData[trainingDataIndex][0:-1])
        # network.getOutput()
        network.forward_propagation(get_desired_output(trainingData[trainingDataIndex]))
        network.backwards_propagation()
    ZETA *= 0.9995
    network.update_zeta(ZETA)
    successRate, correctGuesses= validation(network, validationData)
    successRates.append(successRate)

# plot successrate
plt.plot(range(len(successRates)), successRates, linewidth=1, color='blue', label='successratePerEpoch')
plt.legend()
plt.show()

# sucessrate after final epoch
successRate, correctGuesses = validation(network, validationData)
print(f"the final successrate is {round(successRate, 2)}%                       ")
print(f"The network has correctly guessed {correctGuesses} out of {len(validationData)} datapoints")