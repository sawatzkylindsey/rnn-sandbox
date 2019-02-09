
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

from ml import base as mlbase
from ml import nn as ffnn
from nnwd import geometry
from nnwd.models import Layer, Unit, WeightExplain, WeightVector, LabelWeightVector
from nnwd import nlp
from nnwd import rnn
from pytils.log import setup_logging, user_log
from pytils import adjutant, check


REFERENCES = 32
SHORTLIST_REFERENCES = int(REFERENCES / 3)


def create(corpus, epochs, verbose):
    words, xy_sequences = nlp.corpus_sequences(corpus)

    if verbose:
        for sequence in xy_sequences:
            logging.debug(sequence)

    return words, NeuralNetwork(words, xy_sequences, epochs)


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

    def __init__(self, words, xy_sequences, epoch_threshold):
        self.words = words
        self.lstm = rnn.Rnn(NeuralNetwork.LAYERS, NeuralNetwork.WIDTH, NeuralNetwork.EMBEDDING_WIDTH, words)
        self.xy_sequences = [mlbase.Xy(sequence[:-1], sequence[1:]) for sequence in xy_sequences]
        self.epoch_threshold = epoch_threshold
        self.colour_embeddings = None
        self._background_setup = threading.Thread(target=self._setup)
        self._background_setup.daemon = True
        self._background_setup.start()

    def _setup(self):
        self._train_rnn()
        self._train_predictor()
        self._setup_colour_embeddings()
        user_log.info("Setup complete")

    def _train_rnn(self):
        training_coarse = mlbase.TrainingParameters() \
            .epochs(self.epoch_threshold) \
            .batch(16)
        loss = self.lstm.train(self.xy_sequences, training_coarse)
        logging.debug("train coarse %s" % loss)
        training_fine = mlbase.TrainingParameters() \
            .epochs(self.epoch_threshold) \
            .absolute(loss * 0.75) \
            .batch(1)
        loss = self.lstm.train(self.xy_sequences, training_fine)
        logging.debug("train fine %s" % loss)

    def _train_predictor(self):
        kind_labels = mlbase.Labels(set(NeuralNetwork.MAPPED_KINDS))
        layer_labels = mlbase.Labels(set(range(NeuralNetwork.LAYERS)))
        activation_vector = mlbase.VectorField(NeuralNetwork.WIDTH)
        predictor_input = mlbase.ConcatField([kind_labels, layer_labels, activation_vector])
        predictor_output = mlbase.Labels(self.words.labels())
        hyper_parameters = ffnn.HyperParameters() \
            .width(max(1, int(len(predictor_input) * 0.8)))
        self.predictor = ffnn.Model("predictor", ffnn.HyperParameters().width(int(len(predictor_input) * .8)), predictor_input, predictor_output, mlbase.SINGLE_LABEL)
        self.predictor_data = []

        for sequence in self.xy_sequences:
            stepwise_lstm = self.lstm.stepwise()

            for x in sequence.x:
                result, instruments = stepwise_lstm.step(x, NeuralNetwork.INSTRUMENTS)

                for kind in NeuralNetwork.MAPPED_KINDS:
                    for layer in range(NeuralNetwork.LAYERS):
                        point = tuple(instruments[kind][layer])
                        train_xy = mlbase.Xy((kind, layer, point), result.distribution)
                        self.predictor_data += [train_xy]

        loss = self.predictor.train(self.predictor_data, mlbase.TrainingParameters().epochs(int(self.epoch_threshold / 2)))
        logging.debug("train predictor %s" % loss)

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

    def get_activation_prediction_encoding(self, points):
        if not self.is_setup():
            predictions = {kind: {layer: (nlp.UNKNOWN, 1.0) for layer in range(NeuralNetwork.LAYERS)} for kind in NeuralNetwork.MAPPED_KINDS}
            colours = {kind: {layer: "none" for layer in range(NeuralNetwork.LAYERS)} for kind in NeuralNetwork.MAPPED_KINDS}
            return predictions, colours

        # Maps roughly:
        #   0.00 -> 0.5
        #   0.05 -> 0.7
        #   0.10 -> 0.8
        #   0.20 -> 1.0
        #   1.00 -> 1.0
        likelyness_opacity = lambda x: min(1.0, math.sqrt(x) + 0.5)
        xs = adjutant.flat_map([[(kind, layer, point) for layer, point in subd.items()] for kind, subd in points.items()])
        results = self.predictor.evaluate(xs)
        point_data = {kind: {} for kind in NeuralNetwork.MAPPED_KINDS}
        predictions = {kind: {} for kind in NeuralNetwork.MAPPED_KINDS}

        for i, x in enumerate(xs):
            kind, layer, point = x
            result = results[i]
            predictions[kind][layer] = (result.prediction, likelyness_opacity(result.distribution[result.prediction]))
            # Select the most likely several predictions for this activation vector (kind, layer, point).
            ordered_predictions = [item[0] for item in sorted(result.distribution.items(), key=lambda item: item[1], reverse=True)]
            likely_predictions = []
            proceed = True

            while proceed:
                likely_predictions += [ordered_predictions[len(likely_predictions)]]

                if proceed and len(likely_predictions) >= 2:
                    if len(likely_predictions) > min(5, len(ordered_predictions)):
                        proceed = False
                    else:
                        proceed = result.distribution[likely_predictions[-2]] / 2.0 < result.distribution[likely_predictions[-1]]

            top_predictions = {prediction: result.distribution[prediction] for prediction in likely_predictions}
            distro = {k: 1.0 - v for k, v in nlp.regmax(top_predictions).items()}
            point_data[kind][layer] = [(self.colour_embeddings[k], v) for k, v in distro.items()]

        colours = {}

        for kind, subd in point_data.items():
            colours[kind] = {}

            for layer, data in subd.items():
                # Make the maximum target distance (1.0) one quarter the length of a dimension in the colour space.
                scaler = (255.0 / 4)
                fit, _ = geometry.fit_point([item[0] for item in data], [scaler * item[1] for item in data], epsilon=0.1, visualize=False)
                colours[kind][layer] = "rgb(%d, %d, %d)" % tuple([round(i) for i in fit])

        return predictions, colours

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

        instrument_predictions, instrument_colours = self.get_activation_prediction_encoding(points)
        embedding = WeightVector(instruments["embedding"], colour=self.word_colour(x))
        units = []

        for layer in range(NeuralNetwork.LAYERS):
            remember_gate = WeightVector(instruments["remember_gates"][layer], 0, 1)
            forget_gate = WeightVector(instruments["forget_gates"][layer], 0, 1)
            output_gate = WeightVector(instruments["output_gates"][layer], 0, 1)
            input_hat = WeightVector(instruments["input_hats"][layer], colour=instrument_colours["input_hats"][layer], prediction=instrument_predictions["input_hats"][layer])
            remember = WeightVector(instruments["remembers"][layer], colour=instrument_colours["remembers"][layer], prediction=instrument_predictions["remembers"][layer])
            cell_previous_hat = WeightVector(instruments["cell_previouses"][layer], colour=instrument_colours["cell_previouses"][layer], prediction=instrument_predictions["cell_previouses"][layer])
            forget = WeightVector(instruments["forgets"][layer], colour=instrument_colours["forgets"][layer], prediction=instrument_predictions["forgets"][layer])
            cell = WeightVector(instruments["cells"][layer], colour=instrument_colours["cells"][layer], prediction=instrument_predictions["cells"][layer])
            cell_hat = WeightVector(instruments["cell_hats"][layer], colour=instrument_colours["cell_hats"][layer], prediction=instrument_predictions["cell_hats"][layer])
            output = WeightVector(instruments["outputs"][layer], colour=instrument_colours["outputs"][layer], prediction=instrument_predictions["outputs"][layer])
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

