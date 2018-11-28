
import json
import logging
import pdb
import threading

from nnwd.models import Layer, Unit, WeightExplain, WeightVector, LabelWeightVector
from pytils.log import setup_logging, user_log


class Echo:
    def get(self, data):
        return data


class Words:
    def __init__(self, words):
        self.words = sorted([w for w in words])

    def get(self, data):
        return self.words


class Weights:
    def __init__(self, neural_network):
        self.neural_network = neural_network

    def get(self, data):
        sequence = data["sequence"]

        if sequence == ["<test>"]:
            v = [0, 1, 0.5, -0.5, -1]
            g = [0, 1, 0.5, 0, 1]
            embedding = WeightVector(v)
            units = []

            for layer in range(0, 2):
                remember_gate = WeightVector(g, 0, 1)
                forget_gate = WeightVector(g, 0, 1)
                output_gate = WeightVector(g, 0, 1)
                input_hat = WeightVector(v)
                remember = WeightVector(v)
                cell_previous_hat = WeightVector(v)
                forget = WeightVector(v)
                cell = WeightVector(v)
                cell_hat = WeightVector(v)
                output = WeightVector(v)
                units += [Unit(remember_gate, forget_gate, output_gate, input_hat, remember, cell_previous_hat, forget, cell, cell_hat, output)]

            softmax = LabelWeightVector({"a": 0.5, "b": 0.25, "c": 0.125, "d": 0.0625, "e": 0.0625}, {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}, 5)
            return Layer(embedding, units, softmax, len(sequence) - 1, "<test>", "a")

        return self.neural_network.weights(sequence)


class WeightExplain:
    def __init__(self, neural_network):
        self.neural_network = neural_network

    def get(self, data):
        sequence = data["sequence"]
        name = data["name"][0]
        column = int(data["column"][0])
        return self.neural_network.weight_explain(sequence, name, column)


