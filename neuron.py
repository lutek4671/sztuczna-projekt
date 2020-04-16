from random import uniform
import numpy as np


class Neuron:
    def __init__(self, outputs_number, index):
        self.output_value = 0.0
        self.connections = []
        self.index = index
        for output_index in range(outputs_number):
            self.connections.append({"weight": self.random_weight(), "delta": 0.0})  # ustawienie wag losowo [0-1]

    @staticmethod
    def random_weight():
        return uniform(0, 1)

    @staticmethod
    def tanh_activation(sum_values):
        return np.tanh(sum_values)

    def feed_forward(self, prev_layer):
        sum_values = 0.0
        for neuron_index in range(len(prev_layer)):
            sum_values += prev_layer[neuron_index].output_value * prev_layer[neuron_index].connections[self.index][
                "weight"]
        self.output_value = self.tanh_activation(sum)
