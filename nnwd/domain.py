
import logging
import numpy as np
import pdb
import threading

from nnwd.models import Layer, Unit, WeightVector, LabelWeightVector
from nnwd import nlp
from nnwd import rnn
from pytils.log import setup_logging, user_log


def create(corpus, epochs, verbose):
    words, xy_sequences = nlp.corpus_sequences(corpus)

    if verbose:
        for sequence in xy_sequences:
            logging.debug(sequence)

    return words, xy_sequences, NeuralNetwork(words, xy_sequences, epochs)


class NeuralNetwork:
    LAYERS = 2
    WIDTH = 5
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
    MATRICES = {
        "embedding": ("E", "E_bias"),
        "remember_gates": ("R", "R_bias"),
        "forget_gates": ("F", "F_bias"),
        "output_gates": ("O", "O_bias"),
        "embedding": ("E", "E_bias"),
        "embedding": ("E", "E_bias"),
    }

    def __init__(self, words, xy_sequences, epochs):
        self.lstm = rnn.Rnn(NeuralNetwork.LAYERS, NeuralNetwork.WIDTH, words)
        self.xy_sequences = [[rnn.Xy(t[0], t[1]) for t in sequence] for sequence in xy_sequences]
        self.epochs = epochs
        self._background_training = threading.Thread(target=self._train_test)
        self._background_training.daemon = True
        self._background_training.start()

    def _train_test(self):
        self.lstm.train(self.xy_sequences, self.epochs, True)
        r = self.lstm.test(self.xy_sequences, True)
        logging.debug("test %s" % r)

    def is_training(self):
        return self._background_training.is_alive()

    def weights(self, sequence):
        stepwise_lstm = self.lstm.stepwise()

        for x in sequence:
            result, instruments = stepwise_lstm.step(x, NeuralNetwork.INSTRUMENTS)

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
        return Layer(embedding, units, softmax, len(sequence) - 1)

    def weight_explain(self, name, column):
        if "-" in name:
            self.lstm.probe(name.split("-"))
        else:
            self.lstm.probe(name)

        Y = self.lstm.probe("Y")
        print("Y: %s" % Y)
        Y_bias = self.lstm.probe("Y_bias")
        print("Y_bias: %s" % Y_bias)
        isolated_Y = Y[:, column]
        q = weights * isolated_Y
        print(q)
        return WeightVector(q)

