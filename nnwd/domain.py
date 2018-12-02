
import json
import logging
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pdb
from random import randint
from sklearn.manifold import TSNE
import threading

from nnwd import geometry
from nnwd.models import Layer, Unit, WeightExplain, WeightVector, LabelWeightVector
from nnwd import nlp
from nnwd import rnn
from pytils.log import setup_logging, user_log


REFERENCES = 16


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
        "remember_gate": "R",
        "forget_gate": "F",
        "output_gate": "O",
        "input_hat": "H",
        "softmax": "Y",
    }

    def __init__(self, words, xy_sequences, epochs):
        self.words = words
        self.lstm = rnn.Rnn(NeuralNetwork.LAYERS, NeuralNetwork.WIDTH, words)
        self.xy_sequences = [[rnn.Xy(t[0], t[1]) for t in sequence] for sequence in xy_sequences]
        self.epochs = epochs
        self.colour_embeddings = None
        self._background_setup = threading.Thread(target=self._setup)
        self._background_setup.daemon = True
        self._background_setup.start()

    def _setup(self):
        loss = self.lstm.train(self.xy_sequences, self.epochs, True)
        logging.debug("train %s" % loss)
        accuracy = self.lstm.test(self.xy_sequences, True)
        logging.debug("test %s" % accuracy)
        weights = []
        order = []
        self.word_embeddings = {}

        for word in self.words.labels():
            weight = self.lstm.embed(word)
            weights += [weight]
            order += [word]
            self.word_embeddings[word] = weight

        tsne = TSNE(n_components=3)
        embeddings_3d = tsne.fit_transform(weights)
        minimum = [None, None, None]
        maximum = [None, None, None]

        for embedding in embeddings_3d:
            for i, point in enumerate(embedding):
                if minimum[i] is None or point < minimum[i]:
                    minimum[i] = point

                if maximum[i] is None or point > maximum[i]:
                    maximum[i] = point

        delta = [maximum[i] - minimum[i] for i in range(0, 3)]
        slope = [255.0 / delta[i] for i in range(0, 3)]
        m = lambda i, x: round((slope[i] * x) - (slope[i] * minimum[i]))
        self.colour_embeddings = {order[j]: tuple([m(i, embedding[i]) for i in range(0, 3)]) for j, embedding in enumerate(embeddings_3d)}
        figure = plt.figure()
        axis = figure.add_subplot(111, projection="3d")

        for word, colour_embedding in self.colour_embeddings.items():
            x, y, z = colour_embedding
            axis.scatter(x, y, z, c=[c / 255.0 for c in colour_embedding])
            axis.text(x, y, z, word, zorder=1)

        axis.set_xlabel("r")
        axis.set_ylabel("g")
        axis.set_zlabel("b")
        figure.savefig("word_colour_embedding.png")
        user_log.info("Training complete")
        indices = [i for i in range(0, len(order))]
        reference_indices = set()

        while len(reference_indices) < REFERENCES:
            if len(indices) == 0:
                break

            i = randint(0, len(indices) - 1)
            reference_indices.add(indices[i])
            indices = indices[:i] + indices[i + 1:]

        self.reference_points = [self.word_embeddings[order[i]] for i in reference_indices]
        self.reference_colours = [self.colour_embeddings[order[i]] for i in reference_indices]

    def is_setup(self):
        return not self._background_setup.is_alive()

    def word_colour(self, word):
        if not self.is_setup():
            return "none"

        embedding = self.colour_embeddings[self.words.decode(self.words.encode(word, True))]
        return "rgb(%d, %d, %d)" % embedding

    def fit_colours(self, points):
        if not self.is_setup():
            return ["none"] * len(points)

        point_distances = []
        maximum = None

        for i, point in enumerate(points):
            distances = []

            for reference_point in self.reference_points:
                distance = geometry.distance(point, reference_point)
                distances += [distance]

                if maximum is None or distance > maximum:
                    maximum = distance

            point_distances += [distances]

        # Make the maximum target distance half the size of the colour space.
        scaler = (255.0 / 2) / maximum
        colours = []

        for i in range(0, len(points)):
            fit, t = geometry.fit_point(self.reference_colours, [scaler * d for d in point_distances[i]])
            colours += ["rgb(%d, %d, %d)" % tuple([round(i) for i in fit])]

        return colours

    def weights(self, sequence):
        stepwise_lstm = self.lstm.stepwise()

        for x in sequence:
            x_word = self.words.decode(self.words.encode(x, True))
            result, instruments = stepwise_lstm.step(x, NeuralNetwork.INSTRUMENTS)

        points = []

        for layer in range(0, len(instruments["outputs"])):
            points += [instruments["input_hats"][layer]]
            points += [instruments["remembers"][layer]]
            points += [instruments["cell_previouses"][layer]]
            points += [instruments["forgets"][layer]]
            points += [instruments["cells"][layer]]
            points += [instruments["cell_hats"][layer]]
            points += [instruments["outputs"][layer]]

        instrument_colours = self.fit_colours(points)
        embedding = WeightVector(instruments["embedding"], colour=self.word_colour(x))
        units = []

        for layer in range(0, len(instruments["outputs"])):
            remember_gate = WeightVector(instruments["remember_gates"][layer], 0, 1)
            forget_gate = WeightVector(instruments["forget_gates"][layer], 0, 1)
            output_gate = WeightVector(instruments["output_gates"][layer], 0, 1)
            input_hat = WeightVector(instruments["input_hats"][layer], colour=instrument_colours.pop(0))
            remember = WeightVector(instruments["remembers"][layer], colour=instrument_colours.pop(0))
            cell_previous_hat = WeightVector(instruments["cell_previouses"][layer], colour=instrument_colours.pop(0))
            forget = WeightVector(instruments["forgets"][layer], colour=instrument_colours.pop(0))
            cell = WeightVector(instruments["cells"][layer], colour=instrument_colours.pop(0))
            cell_hat = WeightVector(instruments["cell_hats"][layer], colour=instrument_colours.pop(0))
            output = WeightVector(instruments["outputs"][layer], colour=instrument_colours.pop(0))
            units += [Unit(remember_gate, forget_gate, output_gate, input_hat, remember, cell_previous_hat, forget, cell, cell_hat, output)]

        softmax = LabelWeightVector(result.distribution, result.encoding, NeuralNetwork.WIDTH, lambda word: self.word_colour(word))
        return Layer(embedding, units, softmax, len(sequence) - 1, x_word, result.prediction)

    def weight_explain(self, sequence, name, column):
        weights = self.weights(sequence)

        if len(sequence) > 1:
            weights_previous = self.weights(sequence[:-1])
            embedding_previous = weights_previous.embedding
            outputs_previous = [unit.output for unit in weights_previous.units]
        else:
            zeros = WeightVector([0] * NeuralNetwork.WIDTH)
            embedding_previous = zeros
            outputs_previous = [zeros] * NeuralNetwork.LAYERS

        parts = name.split("-")
        name = parts[0]

        if len(parts) == 2:
            layer = int(parts[1])
        else:
            layer = None

        if name in NeuralNetwork.MATRICES:
            matrix = self.lstm.probe(NeuralNetwork.MATRICES[name], layer)
            bias = self.lstm.probe(NeuralNetwork.MATRICES[name] + "_bias", layer)

            if name.endswith("_gate") or name == "input_hat":
                left = outputs_previous[layer].vector
                left_feed = "output_previous-%d" % layer
            else:
                left = []
                left_feed = None

            if layer == 0:
                right = weights.embedding.vector
                right_feed = "embedding"
            else:
                u = layer - 1 if layer is not None else NeuralNetwork.LAYERS - 1
                right = weights.units[u].output.vector
                right_feed = "output-%d" % (u)

            explain = ((left + right) * matrix[:, column])
            vectors = {
                right_feed: explain[NeuralNetwork.WIDTH:] if len(explain) > NeuralNetwork.WIDTH else explain
            }

            if left_feed is not None:
                vectors[left_feed] = explain[:NeuralNetwork.WIDTH]

            return WeightExplain(vectors, bias[column])
        elif name == "cell":
            left = [0] * NeuralNetwork.WIDTH
            left[column] = weights.units[layer].forget.vector[column]
            left_feed = "forget_hat-%d" % layer
            right = [0] * NeuralNetwork.WIDTH
            right[column] = weights.units[layer].remember.vector[column]
            right_feed = "remember_hat-%d" % layer
            return WeightExplain({left_feed: left, right_feed: right}, 0)
        elif name == "forget":
            explain = [0] * NeuralNetwork.WIDTH
            explain[column] = weights.units[layer].cell_previous_hat.vector[column] * weights.units[layer].forget_gate.vector[column]
            return WeightExplain({"cell_previous_hat-%d" % layer: explain}, 0)
        elif name == "remember":
            explain = [0] * NeuralNetwork.WIDTH
            explain[column] = weights.units[layer].input_hat.vector[column] * weights.units[layer].remember_gate.vector[column]
            return WeightExplain({"input_hat-%d" % layer: explain}, 0)
        elif name == "output":
            explain = [0] * NeuralNetwork.WIDTH
            explain[column] = weights.units[layer].cell_hat.vector[column] * weights.units[layer].output_gate.vector[column]
            return WeightExplain({"cell_hat-%d" % layer: explain}, 0)
        elif name == "remember_hat":
            explain = [0] * NeuralNetwork.WIDTH
            explain[column] = weights.units[layer].remember.vector[column]
            return WeightExplain({"remember-%d" % layer: explain}, 0)
        elif name == "forget_hat":
            explain = [0] * NeuralNetwork.WIDTH
            explain[column] = weights.units[layer].forget.vector[column]
            return WeightExplain({"forget-%d" % layer: explain}, 0)
        elif name == "cell_hat":
            explain = [0] * NeuralNetwork.WIDTH
            explain[column] = weights.units[layer].cell.vector[column]
            return WeightExplain({"cell-%d" % layer: explain}, 0)
        elif name == "cell_previous_hat":
            explain = [0] * NeuralNetwork.WIDTH
            explain[column] = weights.units[layer].cell_previous_hat.vector[column]
            return WeightExplain({"cell_previous-%d" % layer: explain}, 0)
        else:
            raise ValueError("unknown: %s - %s" % (name, column))

    def addition(self, a, b):
        return [a[i] + b[i] for i in range(0, len(a))]

    def multiplication(self, a, b):
        return [a[i] * b[i] for i in range(0, len(a))]

    def sigmoid(self, value):
        return 1.0 / (1.0 + math.exp(-value))

    def tanh(self, value):
        part = math.exp(-2 * value)
        return (1.0 - part) / (1.0 + part)

