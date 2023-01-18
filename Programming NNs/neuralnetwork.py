import random
random.seed(0)
from neuron import InitialNeuron, Neuron
import copy

def transform_array(array):
    transformedArray = []
    tempArray = []
    for xaxis in range(len(array[0])):
        for yaxis in range(len(array)):
            tempArray.append(array[yaxis][xaxis])
        transformedArray.append(tempArray)
        tempArray = []
    return transformedArray

class NeuralNetwork:
    def __init__(self):
        self.layers = [[]]
        self.outputDeltas = []
        self.outputWeights = []
        
    def set_input_layer(self, inputData): 
        self.layers[0] = []
        for dataPoint in inputData:
            self.layers[0].append(InitialNeuron(dataPoint))   

        for layersIndex in range(1, len(self.layers)):
            for neuron in self.layers[layersIndex]:
                neuron.set_input_neurons(self.layers[layersIndex-1])
        
    def add_layer(self, neuronAmmount, zeta=0.1, weightMin=-1.0, weightMax=1.0, biasMin=-1.0, biasMax=1.0):
        layer = []
        for notUsed in range(neuronAmmount):
            startingWeights = []
            for notUsed in range(len(self.layers[-1])):
                startingWeights.append(random.uniform(weightMin, weightMax))
            layer.append(Neuron(self.layers[-1], startingWeights, random.uniform(biasMin, biasMax), zeta))
        self.layers.append(layer)



    def forward_propagation(self, desiredOutputs):
        self.outputDeltas = []
        self.outputWeights = []

        # Delta rule for output layer
        for neuronIndex, neuron in enumerate(self.layers[-1]):  
            neuron.get_output()
            delta, weights = neuron.output_delta_rule(desiredOutputs[neuronIndex])
            self.outputDeltas.append(delta)
            self.outputWeights.append(weights)  
        self.outputWeights = transform_array(self.outputWeights)
      

        # Delta rule for middle layer(s)
        for layerIndex in range(len(self.layers)-2, 0, -1):
            oldOutputDeltas = copy.deepcopy(self.outputDeltas)
            oldOutputWeights = copy.deepcopy(self.outputWeights)
            self.outputDeltas = []
            self.outputWeights = []
            for neuronIndex, neuron in enumerate(self.layers[layerIndex]):
                delta, weights = neuron.delta_rule(oldOutputDeltas, oldOutputWeights[neuronIndex])
                self.outputDeltas.append(delta)
                self.outputWeights.append(weights) 
            self.outputWeights = transform_array(self.outputWeights)

    
    def backwards_propagation(self):
        for layerIndex in range(len(self.layers)-1, 0, -1):
            # print(f"layerIndex: {layerIndex}")
            for neuron in self.layers[layerIndex]:
                neuron.update()
    
    def get_output(self):
        # print("1")
        outputs = []
        
        for neuron in self.layers[-1]: 
            outputs.append(neuron.get_output())
        return outputs

    def update_zeta(self, newZeta):
        for layerIndex in range(1, len(self.layers)-1):
            for neuron in self.layers[layerIndex]:  
                neuron.set_zeta(newZeta)
