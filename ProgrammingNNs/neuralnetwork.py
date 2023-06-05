import random
random.seed(0)
from neuron import InitialNeuron, Neuron
import copy

def transform_array(array):
    """
    This function transposes a two dimensional array

    :param array: List to be transposed
    :return transformed_array: Transposed array
    """
    transformed_array = []
    tmp_array = []
    for xaxis in range(len(array[0])):
        for yaxis in range(len(array)):
            tmp_array.append(array[yaxis][xaxis])
        transformed_array.append(tmp_array)
        tmp_array = []
    return transformed_array


class NeuralNetwork:
    def __init__(self):
        """
        This function initializes the neural network.
        """
        self.layers = [[]]
        self.output_deltas = []
        self.output_weights = []
        
    def set_input_layer(self, input_data):
        """
        This function sets the input layer of the neural network.
        It takes in a list of input data, and creates a neuron for each datapoint.
        It then sets the input neurons for each neuron in the next layer.

        :param input_data: A list of input data
        """ 
        self.layers[0] = []
        for datapoint in input_data:
            self.layers[0].append(InitialNeuron(datapoint))   

        for layers_index in range(1, len(self.layers)):
            for neuron in self.layers[layers_index]:
                neuron.set_input_neurons(self.layers[layers_index-1])
        
    def add_layer(self, amount_of_neurons, zeta=0.1, weight_min=-1.0, weight_max=1.0, bias_min=-1.0, bias_max=1.0):
        """
        Adds a layer to the neural network

        :param amount_of_neurons: The amount of neurons in the layer.
        :param zeta: The learning rate of the neurons in the layer.
        :param weight_min: The minimum weight of the neurons in the layer.
        :param weight_max: The maximum weight of the neurons in the layer.
        :param bias_min: The minimum bias of the neurons in the layer.
        :param bias_max: The maximum bias of the neurons in the layer.
        """
        layer = []
        for not_used in range(amount_of_neurons):
            starting_weights = []
            for not_used in range(len(self.layers[-1])):
                starting_weights.append(random.uniform(weight_min, weight_max))
            layer.append(Neuron(self.layers[-1], starting_weights, random.uniform(bias_min, bias_max), zeta))
        self.layers.append(layer)



    def forward_propagation(self, desired_outputs):
        """
        This function performs the forward propagation of the neural network.
        It calculates the output of each neuron in the network and the error
        of the output layer.
        It also calculates the deltas and weights for each neuron in the network.
        
        :param desired_outputs: Given list with desired_outputs
        """

        self.output_deltas = []
        self.output_weights = []

        # Delta rule for output layer
        for neuron_index, neuron in enumerate(self.layers[-1]):  
            neuron.get_output()
            delta, weights = neuron.output_delta_rule(desired_outputs[neuron_index])
            self.output_deltas.append(delta)
            self.output_weights.append(weights)  
        self.output_weights = transform_array(self.output_weights)
      

        # Delta rule for middle layer(s)
        for layer_index in range(len(self.layers)-2, 0, -1):
            old_output_deltas = copy.deepcopy(self.output_deltas)
            old_output_weights = copy.deepcopy(self.output_weights)
            self.output_deltas = []
            self.output_weights = []
            for neuron_index, neuron in enumerate(self.layers[layer_index]):
                delta, weights = neuron.delta_rule(old_output_deltas, old_output_weights[neuron_index])
                self.output_deltas.append(delta)
                self.output_weights.append(weights) 
            self.output_weights = transform_array(self.output_weights)

    
    def backwards_propagation(self):
        """
        This function is used to update the weight of the neurons in the network.
        """
        for layer_index in range(len(self.layers)-1, 0, -1):
            # print(f"layerIndex: {layerIndex}")
            for neuron in self.layers[layer_index]:
                neuron.update()
    
    def get_output(self):
        """
        This function returns the output of the last layer of the neural network.

        :return outputs: A list with all outputs of the last layer of the neural network.
        """
        outputs = []
        
        for neuron in self.layers[-1]: 
            outputs.append(neuron.get_output())
        return outputs

    def update_zeta(self, new_zeta):
        """
        This function updates the zeta value for all neurons in the network.

        :param new_zeta: The new zeta value to be used.
        """
        for layer_index in range(1, len(self.layers)-1):
            for neuron in self.layers[layer_index]:  
                neuron.set_zeta(new_zeta)
