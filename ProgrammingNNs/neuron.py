import math


class InitialNeuron:
    """
    This is a class for the initial neuron.
    """
    def __init__(self, value):
        self.value = value

    def get_output(self):
        return self.value

    def get_a(self):
        return self.value

    def __str__(self):
        return str(self.get_output())



class Neuron:
    """
    This is a class for a single neuron.
    """
    def __init__(self, input_neurons, input_weights, bias, ZETA):
        """
        This function initializes the neuron with the following parameters:
        
        :param input_neurons: The input neurons to the neuron
        :param input_weights: The weights of the input neurons
        :param bias: The bias of the neuron
        :param ZETA: The activation function of the neuron
        """
        self.input_neurons = input_neurons
        self.input_weights = input_weights
        self.bias = bias
        self.ZETA = ZETA

        self.z = None
        self.a = None

        
    def set_input_neurons(self, input_neurons):
        """
        Sets the input neurons of the network.

        :param input_neurons: A list of input neurons.
        """
        self.input_neurons = input_neurons


    def get_output(self):
        """ 
        This function calculates the output of the neuron.

        :return: The output of the neuron.        
        """

        input_sum = 0
        for index in range(0, len(self.input_neurons)):
            input_sum += self.input_neurons[index].get_output() * self.input_weights[index]

        self.z = self.bias + input_sum
        self.a = math.tanh(self.z)
        return self.a


    def delta_rule(self, deltas, weights):
        """
        This function is used to calculate the delta value for a hidden layer neuron.
        It takes in the deltas of the neurons in the next layer and the weights of the connections between the neurons in the next layer and the current neuron.
        It then calculates the delta value for the current neuron and returns it.

        :param deltas: A list of the deltas of the neurons in the next layer.
        :param weights: A list of the weights of the connections between the neurons in the next layer and the current neuron.
        :return delta: The delta value for the current neuron.
        :return input_weights: A list of the weights of the connections between the neurons in the previous layer and the current neuron.
        """
        output_sum = 0
        for index in range(0, len(deltas)):
            output_sum += deltas[index] * weights[index]
        self.delta = (1 - math.tanh(self.z)**2 ) * output_sum
        return self.delta, self.input_weights


    def output_delta_rule(self, desired_output):
        """
        This function calculates the delta value for a neuron.
        It takes the desired output as an argument.
        It returns the delta value and the input weights.

        :param desired_output: The desired output of the neuron.
        :return delta: The delta value of the neuron
        :return input_weights: The input weights of the neuron.
        """
        self.delta = (1 - math.tanh(self.z)**2) * (desired_output - self.a)
        return self.delta, self.input_weights


    def update(self):
        """
        Updates the weights and bias of the neuron.
        """
        for index in range(0, len(self.input_weights)):
            self.input_weights[index] += (self.ZETA * self.delta * self.input_neurons[index].get_a())
        self.bias += self.ZETA * self.delta


    def get_a(self):
        """
        Get output of neuron.
        """
        return self.a

    def set_zeta(self, zeta):
        """
        Set the activation function of the neuron.

        :param zeta: The activation function of the neuron
        """
        self.ZETA = zeta



