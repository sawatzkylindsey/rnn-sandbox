
import logging
import pdb
import threading

from pytils.log import setup_logging, user_log
from nnwd.models import Layer, Unit, WeightVector, LabelWeightVector
from nnwd import rnn


class Echo:
    def get(self, data):
        return data


class Words:
    def __init__(self, words):
        self.words = sorted([w for w in words])

    def get(self, data):
        return self.words


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
    LAYERS = 2
    WIDTH = 5

    def __init__(self, words, xy_sequences, epochs):
        self.neural_network = rnn.Rnn(NeuralNetwork.LAYERS, NeuralNetwork.WIDTH, words)
        self.xy_sequences = [[rnn.Xy(t[0], t[1]) for t in sequence] for sequence in xy_sequences]
        self.epochs = epochs
        self._background_training = threading.Thread(target=self._train_test)
        self._background_training.daemon = True
        self._background_training.start()

    def _train_test(self):
        self.neural_network.train(self.xy_sequences, self.epochs, True)
        r = self.neural_network.test(self.xy_sequences, True)
        logging.debug("test %s" % r)

    def get(self, data):
        stepwise_rnn = self.neural_network.stepwise()

        for x in data["sequence"]:
            result, instruments = stepwise_rnn.step(x, NeuralNetwork.INSTRUMENTS)

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

        softmax = LabelWeightVector(result.distribution, NeuralNetwork.WIDTH)
        return Layer(embedding, units, softmax, len(data["sequence"]) - 1)

