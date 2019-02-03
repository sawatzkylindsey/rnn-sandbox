
import itertools
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
from pytils import adjutant, check


REFERENCES = 32
SHORTLIST_REFERENCES = int(REFERENCES / 3)


def create(corpus, epochs, loss, verbose):
    words, xy_sequences = nlp.corpus_sequences(corpus)

    if verbose:
        for sequence in xy_sequences:
            logging.debug(sequence)

    return words, xy_sequences, NeuralNetwork(words, xy_sequences, epochs, loss)


class NeuralNetwork:
    LAYERS = 2
    WIDTH = 10
    EMBEDDING_WIDTH = 5
    OUTPUT_WIDTH = 7
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
    MAPPED_KINDS = [
        "input_hats",
        "remembers",
        "cell_previouses",
        "forgets",
        "cells",
        "cell_hats",
        "outputs",
    ]

    def __init__(self, words, xy_sequences, epoch_threshold, loss_threshold):
        self.words = words
        self.lstm = rnn.Rnn(NeuralNetwork.LAYERS, NeuralNetwork.WIDTH, NeuralNetwork.EMBEDDING_WIDTH, words)
        self.xy_sequences = [[rnn.Xy(t[0], t[1]) for t in sequence] for sequence in xy_sequences]
        self.epoch_threshold = epoch_threshold
        self.loss_threshold = loss_threshold
        self.colour_embeddings = None
        self._background_setup = threading.Thread(target=self._setup)
        self._background_setup.daemon = True
        self._background_setup.start()

    def _setup(self):
        loss = self.lstm.train(self.xy_sequences, self.epoch_threshold, self.loss_threshold, True)
        logging.debug("train %s" % loss)
        self._setup_reference_points()
        self._setup_colour_embeddings()
        user_log.info("Setup complete")

    def _setup_reference_points(self):
        self.reference_points = {kind: {layer: [] for layer in range(NeuralNetwork.LAYERS)} for kind in NeuralNetwork.MAPPED_KINDS}
        self.reference_distribution = {kind: {layer: {} for layer in range(NeuralNetwork.LAYERS)} for kind in NeuralNetwork.MAPPED_KINDS}
        self.maximum_distance = {kind: {layer: 0 for layer in range(NeuralNetwork.LAYERS)} for kind in NeuralNetwork.MAPPED_KINDS}
        firsts = {kind: {} for kind in NeuralNetwork.MAPPED_KINDS}

        for xy_sequence in self.xy_sequences:
            stepwise_lstm = self.lstm.stepwise()

            for x in xy_sequence:
                x_word = self.words.decode(self.words.encode(x, True))
                result, instruments = stepwise_lstm.step(x, NeuralNetwork.INSTRUMENTS)

                for kind in NeuralNetwork.MAPPED_KINDS:
                    for layer in range(NeuralNetwork.LAYERS):
                        point = tuple(instruments[kind][layer])
                        self.reference_points[kind][layer] += [point]

                        if layer not in firsts[kind]:
                            firsts[kind][layer] = point
                        else:
                            # Establish a baseline maximum_distance
                            distance = geometry.distance(firsts[kind][layer], point)

                            if distance > self.maximum_distance[kind][layer]:
                                self.maximum_distance[kind][layer] = distance

                        if point not in self.reference_distribution[kind][layer]:
                            self.reference_distribution[kind][layer][point] = [result.distribution]
                        else:
                            self.reference_distribution[kind][layer][point] += [result.distribution]

    def _setup_colour_embeddings(self):
        weights = []
        order = []
        self.word_embeddings = {}

        for word in self.words.labels():
            weight = self.lstm.embed(word)
            weights += [weight]
            order += [word]
            self.word_embeddings[word] = weight

        assert len(self.word_embeddings) >= SHORTLIST_REFERENCES, "not enough words (%d < %d" % (len(self.word_embeddings), SHORTLIST_REFERENCES)
        self.colour_embeddings = self.find_colour_embeddings(weights, order)

    def find_colour_embeddings(self, weights, order):
        colour_embeddings = {}

        for perplexity in [5, 10, 20, 30, 50]:
            tsne = TSNE(n_components=3, perplexity=perplexity)
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
            colour_embeddings[perplexity] = {order[j]: tuple([m(i, embedding[i]) for i in range(0, 3)]) for j, embedding in enumerate(embeddings_3d)}
            figure = plt.figure()
            axis = figure.add_subplot(111, projection="3d")

            for word, colour_embedding in colour_embeddings[perplexity].items():
                x, y, z = colour_embedding
                axis.scatter(x, y, z, c=[[c / 255.0 for c in colour_embedding]])
                axis.text(x, y, z, word, zorder=1)
                logging.debug("colour_embedding '%s': %s -> %s" % (word, self.word_embeddings[word], colour_embedding))

            axis.set_xlabel("Red")
            axis.set_ylabel("Green")
            axis.set_zlabel("Blue")

            if perplexity < 10:
                figure.savefig("word_colour_embedding-0%d.png" % perplexity)
            else:
                figure.savefig("word_colour_embedding-%d.png" % perplexity)

        choice = input("choice (05, 10, 20, 30, 50): ")
        return colour_embeddings[int(choice)]

    def is_setup(self):
        return not self._background_setup.is_alive()

    def word_colour(self, word):
        if not self.is_setup():
            return "none"

        embedding = self.colour_embeddings[self.words.decode(self.words.encode(word, True))]
        return "rgb(%d, %d, %d)" % embedding

    def interpolate_distributions(self, points, distribution):
        if not self.is_setup():
            return distribution

        interpolate = lambda k, l, d: 1.0 / (1 + math.exp((d * 10 / self.maximum_distance[k][l]) - 2.0))
        interpolations = {kind: {} for kind in NeuralNetwork.MAPPED_KINDS}
        logging.debug("base: %s" % adjutant.dict_as_str(distribution, use_key=False))

        for kind, subd in points.items():
            for layer, point in subd.items():
                factor = interpolate(kind, layer, 0)
                interpolation = {k: factor * v for k, v in distribution.items()}
                logging.debug("initial: %s-%d: %s" % (kind, layer, adjutant.dict_as_str(interpolation, use_key=False)))

                for reference_point in self.reference_points[kind][layer]:
                    distance = geometry.distance(point, reference_point)

                    if distance > self.maximum_distance[kind][layer]:
                        self.maximum_distance[kind][layer] = distance

                    factor = interpolate(kind, layer, distance)

                    if factor >= 0.5:
                        for value in self.reference_distribution[kind][layer][reference_point]:
                            for k, v in value.items():
                                interpolation[k] += factor * v

                interpolations[kind][layer] = nlp.softmax(interpolation)
                logging.debug("final: %s-%d: %s" % (kind, layer, adjutant.dict_as_str(interpolations[kind][layer], use_key=False)))

        return interpolations

    def fit_colours(self, points, distribution):
        if not self.is_setup():
            return ["none"] * len(points)

        interpolations = self.interpolate_distributions(points, distribution)
        point_data = {kind: {} for kind in NeuralNetwork.MAPPED_KINDS}

        for kind, subd in points.items():
            for layer, point in subd.items():
                # Select the lowest several distances and put them into a map keyed by the reference_word_colour
                # (reference_word_colour -> distance to the equivalent point in the high dimensional space).
                ordered_predictions = [item[0] for item in sorted(interpolations[kind][layer].items(), key=lambda item: item[1], reverse=True)]
                references = []
                proceed = True

                while proceed:
                    references += [ordered_predictions[len(references)]]

                    if proceed and len(references) >= 2:
                        if len(references) > min(5, len(ordered_predictions)):
                            proceed = False
                        else:
                            proceed = interpolations[kind][layer][references[-2]] / 2.0 < interpolations[kind][layer][references[-1]]

                predictions = {reference: interpolations[kind][layer][reference] for reference in references}
                distro = {k: 1.0 - v for k, v in nlp.regmax(predictions).items()}
                logging.debug("refs: %s -> %s" % (adjutant.dict_as_str(predictions, use_key=False), adjutant.dict_as_str(distro, use_key=False)))
                point_data[kind][layer] = [(self.colour_embeddings[k], v) for k, v in distro.items()]
                logging.debug("point_data: %s-%d: %s" % (kind, layer, point_data[kind][layer]))

        colours = {}

        for kind, subd in point_data.items():
            colours[kind] = {}

            for layer, data in subd.items():
                # Make the maximum target distance (1.0) one quarter the length of a dimension in the colour space.
                scaler = (255.0 / 4)
                fit, _ = geometry.fit_point([item[0] for item in data], [item[1] for item in data], epsilon=0.1, visualize=False)
                colours[kind][layer] = "rgb(%d, %d, %d)" % tuple([round(i) for i in fit])

        logging.debug("colours: %s" % colours)
        return colours

    def weights(self, sequence):
        stepwise_lstm = self.lstm.stepwise()

        for x in sequence:
            x_word = self.words.decode(self.words.encode(x, True))
            result, instruments = stepwise_lstm.step(x, NeuralNetwork.INSTRUMENTS)

        points = {}

        for kind in NeuralNetwork.MAPPED_KINDS:
            points[kind] = {}

            for layer in range(NeuralNetwork.LAYERS):
                points[kind][layer] = instruments[kind][layer]

        instrument_colours = self.fit_colours(points, result.distribution)
        embedding = WeightVector(instruments["embedding"], colour=self.word_colour(x))
        units = []

        for layer in range(0, len(instruments["outputs"])):
            remember_gate = WeightVector(instruments["remember_gates"][layer], 0, 1)
            forget_gate = WeightVector(instruments["forget_gates"][layer], 0, 1)
            output_gate = WeightVector(instruments["output_gates"][layer], 0, 1)
            input_hat = WeightVector(instruments["input_hats"][layer], colour=instrument_colours[kind][layer])
            remember = WeightVector(instruments["remembers"][layer], colour=instrument_colours[kind][layer])
            cell_previous_hat = WeightVector(instruments["cell_previouses"][layer], colour=instrument_colours[kind][layer])
            forget = WeightVector(instruments["forgets"][layer], colour=instrument_colours[kind][layer])
            cell = WeightVector(instruments["cells"][layer], colour=instrument_colours[kind][layer])
            cell_hat = WeightVector(instruments["cell_hats"][layer], colour=instrument_colours[kind][layer])
            output = WeightVector(instruments["outputs"][layer], colour=instrument_colours[kind][layer])
            units += [Unit(remember_gate, forget_gate, output_gate, input_hat, remember, cell_previous_hat, forget, cell, cell_hat, output)]

        softmax = LabelWeightVector(result.distribution, result.encoding, NeuralNetwork.OUTPUT_WIDTH, lambda word: self.word_colour(word))
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

