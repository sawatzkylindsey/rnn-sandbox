
import logging
import pdb
import threading

from pytils.log import setup_logging, user_log
from nnwd.models import Layer, Unit, WeightVector, LabelWeightVector
from nnwd import rnn


class Echo:
    def get(self, data):
        return data


class NeuralNetwork:
    INSTRUMENTS = [
        "embedding",
        "remember_gates",
        "forget_gates",
        "output_gates",
        "input_hats",
        "remembers",
        "cell_previouses",
        "forgets",
        "cells",
        "cell_hats",
        "outputs",
    ]

    def __init__(self, words, xy_sequences):
        self.neural_network = rnn.Rnn(1, 5, words)
        self._background_training = threading.Thread(target=self.neural_network.train, args=([[rnn.Xy(t[0], t[1]) for t in sequence] for sequence in xy_sequences], 100, True))
        self._background_training.daemon = True
        self._background_training.start()

    def get(self, data):
        self._background_training.join()
        stepwise_rnn = self.neural_network.stepwise()
        result, instruments = stepwise_rnn.step("the", NeuralNetwork.INSTRUMENTS)
        embedding = WeightVector(instruments["embedding"])
        units = []

        for layer in range(0, len(instruments["outputs"])):
            remember_gate = WeightVector(instruments["remember_gates"][layer], -1, 1)
            forget_gate = WeightVector(instruments["forget_gates"][layer], -1, 1)
            output_gate = WeightVector(instruments["output_gates"][layer], -1, 1)
            input_hat = WeightVector(instruments["input_hats"][layer])
            remember = WeightVector(instruments["remembers"][layer])
            cell_previous = WeightVector(instruments["cell_previouses"][layer])
            forget = WeightVector(instruments["forgets"][layer])
            cell = WeightVector(instruments["cells"][layer])
            cell_hat = WeightVector(instruments["cell_hats"][layer])
            output = WeightVector(instruments["outputs"][layer])
            units += [Unit(remember_gate, forget_gate, output_gate, input_hat, remember, cell_previous, forget, cell, cell_hat, output)]

        softmax = LabelWeightVector(result.distribution)
        return Layer(embedding, units, softmax)

