import math


class InitialNeuron:
    def __init__(self, value):
        self.value = value

    def get_output(self):
        return self.value

    def get_a(self):
        return self.value

    def __str__(self):
        return str(self.get_output())



class Neuron:
    def __init__(self, input_neurons, input_weights, bias, ZETA):
        self.input_neurons = input_neurons
        self.input_weights = input_weights
        self.bias = bias
        self.ZETA = ZETA

        self.z = None
        self.a = None

        
    def set_input_neurons(self, input_neurons):
        self.input_neurons = input_neurons


    def get_output(self):
        input_sum = 0
        for index in range(0, len(self.input_neurons)):
            input_sum += self.input_neurons[index].get_output() * self.input_weights[index]

        self.z = self.bias + input_sum
        self.a = math.tanh(self.z)
        return self.a


    def delta_rule(self, deltas, weights):
        output_sum = 0
        for index in range(0, len(deltas)):
            output_sum += deltas[index] * weights[index]
        self.delta = (1 - math.tanh(self.z)**2 ) * output_sum
        return self.delta, self.input_weights


    def output_delta_rule(self, desired_output):
        self.delta = (1 - math.tanh(self.z)**2) * (desired_output - self.a)
        return self.delta, self.input_weights


    def update(self):
        for index in range(0, len(self.input_weights)):
            self.input_weights[index] += (self.ZETA * self.delta * self.input_neurons[index].get_a())
        self.bias += self.ZETA * self.delta


    def get_a(self):
        return self.a

    def set_zeta(self, zeta):
        self.ZETA = zeta



