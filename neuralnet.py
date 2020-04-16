from neuron import Neuron


class Net:
    def __init__(self, topology):
        self.layers = []
        self.layers_num = len(topology)
        for layer_index in range(self.layers_num):
            self.layers.append([])
            for neuron_index in range(topology[layer_index] + 1):  # dodanie neuronu bedacego biasem
                outputs_number = 0 if layer_index == len(topology) - 1 else topology[
                                                                                layer_index + 1] + 1  # +1 uwzglednienie biasu
                self.layers[-1].append(Neuron(outputs_number, neuron_index))

    def feed_forward(self, input_values):
        if len(input_values) != len(
                self.layers[0]) - 1:  # liczba inputow musi byc taka sama jak liczba neuronow wejsciowych
            raise IndexError

        for neuron_index in range(len(input_values)):  # wpisanie inputu do warsty wejsciowej
            self.layers[0][neuron_index].output_value = input_values[neuron_index]

        for layer_index in range(1, len(self.layers)):
            prev_layer = self.layers[layer_index - 1]

            for neuron_index in range(len(self.layers[layer_index])):
                self.layers[layer_index][neuron_index].feed_forward(prev_layer)

    def back_propagate(self, target_values):
        pass
