
import itertools
import json
import logging
import math
#import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nltk.tokenize import word_tokenize
import numpy as np
import pdb
import random
from sklearn.manifold import TSNE
import threading

from ml import base as mlbase
from ml import nlp
from ml import nn as ffnn
from nnwd import geometry
from nnwd.models import Layer, Unit, WeightExplain, HiddenState, LabelDistribution
from nnwd import rnn
from pytils.log import setup_logging, user_log
from pytils import adjutant, check


def create(reviews, epochs, verbose):
    xys = []
    vocabulary = set()
    classes = set()

    for review in reviews:
        text = word_tokenize(review["text"])

        if len(text) <= 100:
            stars = int(review["stars"])
            assert stars == review["stars"], "%s != %s" % (stars, review["stars"])
            stars = str(stars)
            xys.append((text, stars))
            classes.add(stars)

            for word in text:
                vocabulary.add(word)

    if verbose:
        for xy in xys:
            logging.debug(xy)

    random.shuffle(xys)
    split_1 = int(len(xys) * 0.8)
    split_2 = split_1 + int(len(xys) * 0.1)
    logging.debug("data splits: 0,%d,%d,%d" % (split_1, split_2, len(xys)))
    train_xys = xys[:split_1]
    validation_xys = xys[split_1:split_2]
    test_xys = xys[split_2:]
    logging.debug("datas (train, validation, test): %d, %d, %d" % (len(train_xys), len(validation_xys), len(test_xys)))
    words = mlbase.Labels(vocabulary.union(set([mlbase.BLANK])), unknown=nlp.UNKNOWN)
    sentiments = mlbase.Labels(classes)
    return words, NeuralNetwork(words, sentiments, train_xys, epochs, validation_xys, test_xys)


class NeuralNetwork:
    LAYERS = 2
    HIDDEN_WIDTH = 50
    EMBEDDING_WIDTH = 50
    OUTPUT_WIDTH = 7
    HIDDEN_REDUCTION = 10
    EMBEDDING_REDUCTION = 10
    TOP_PREDICTIONS = 2
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
    LSTM_PARTS = [
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
    SINGULAR = {
        "remember_gates": "remember_gate",
        "forget_gates": "forget_gate",
        "output_gates": "output_gate",
        "input_hats": "input_hat",
        "remembers": "remember",
        "cell_previouses": "cell_previous",
        "forgets": "forget",
        "cells": "cell",
        "cell_hats": "cell_hat",
        "outputs": "output",
    }

    def __init__(self, words, sentiments, train_xys, epoch_threshold, validation_xys, test_xys):
        self.words = check.check_instance(words, mlbase.Labels)
        self.sentiments = check.check_instance(sentiments, mlbase.Labels)
        self.lstm = rnn.Rnn(NeuralNetwork.LAYERS, NeuralNetwork.HIDDEN_WIDTH, NeuralNetwork.EMBEDDING_WIDTH, words, sentiments)
        self.train_xys = [mlbase.Xy(*pair) for pair in train_xys]
        self.validation_xys = [mlbase.Xy(*pair) for pair in validation_xys]
        self.test_xys = [mlbase.Xy(*pair) for pair in test_xys]
        self.epoch_threshold = epoch_threshold
        self._background_setup = threading.Thread(target=self._setup)
        self._background_setup.daemon = True
        self._background_setup.start()

    def _setup(self):
        self._train_rnn()
        self._setup_colour_embeddings()
        self._train_predictor()
        # Also set inside self._train_predictor().
        self.setup_complete = True
        user_log.info("Setup complete")

    def is_setup(self):
        return self.setup_complete
        #return not self._background_setup.is_alive()

    def _train_rnn(self):
        accuracy_validation = 0.0
        batch = 256
        arc = 0

        while accuracy_validation < 0.8 and arc < 20:
            arc += 1
            training_coarse = mlbase.TrainingParameters() \
                .batch(max(8, int(batch / arc))) \
                .epochs(self.epoch_threshold) \
                .convergence(False)
            loss = self.lstm.train(self.train_xys, training_coarse)
            accuracy_validation = self.lstm.test(self.validation_xys)
            logging.debug("train lstm arc %d: (loss, accuracy) (%s, %s)" % (arc, loss, accuracy_validation))

        accuracy_test = self.lstm.test(self.test_xys)
        logging.debug("(v, t): (%s, %s)" % (accuracy_validation, accuracy_test))
        #training_fine = mlbase.TrainingParameters() \
        #    .epochs(self.epoch_threshold) \
        #    .absolute(loss * 0.5) \
        #    .batch(2)
        #loss = self.lstm.train(self.train_xys, training_fine)
        #logging.debug("train lstm fine %s" % loss)

    def _train_predictor(self):
        part_labels = mlbase.Labels(set(NeuralNetwork.LSTM_PARTS))
        layer_labels = mlbase.Labels(set(range(NeuralNetwork.LAYERS)))
        #distance_field = mlbase.IntegerField()
        hidden_vector = mlbase.VectorField(NeuralNetwork.HIDDEN_WIDTH)
        predictor_input = mlbase.ConcatField([part_labels, layer_labels, hidden_vector])
        #predictor_input = mlbase.ConcatField([part_labels, layer_labels, distance_field, hidden_vector])
        predictor_output = mlbase.Labels(set(self.sentiments.labels()))
        hyper_parameters = ffnn.HyperParameters() \
            .width(max(1, int(len(predictor_input) * 0.75)))
        self.predictor = ffnn.Model("predictor", hyper_parameters, predictor_input, predictor_output, mlbase.SINGLE_LABEL)
        self.predictor_data = []

        for xy in self.train_xys:
            stepwise_lstm = self.lstm.stepwise(False)
            #xs = []

            for i, word in enumerate(xy.x):
                result, instruments = stepwise_lstm.step(word, NeuralNetwork.INSTRUMENTS)
                #xs_next = []
                #distance = len(xy.x) - i - 1

                #for x in xs:
                #    x_moved = (x[0], x[1], x[2] + 1, x[3])
                #    xs_next += [x_moved]
                #    train_xy = mlbase.Xy(x_moved, result.distribution)
                #    self.predictor_data += [train_xy]

                for part in NeuralNetwork.LSTM_PARTS:
                    for layer in range(NeuralNetwork.LAYERS):
                        point = tuple(instruments[part][layer])
                        x = (part, layer, point)
                        #x = (part, layer, distance, point)
                        #xs_next += [x]
                        train_xy = mlbase.Xy(x, result.distribution)
                        self.predictor_data += [train_xy]

                #xs = xs_next

            #for x in xs:
            #    train_xy = mlbase.Xy(x, result.distribution)
            #    self.predictor_data += [train_xy]

        self.setup_complete = True
        training_coarse = mlbase.TrainingParameters() \
            .epochs(max(1, int(self.epoch_threshold / 2))) \
            .batch(16)
        loss = self.predictor.train(self.predictor_data, training_coarse)
        logging.debug("train predictor coarse %s" % loss)

    def _setup_colour_embeddings(self):
        self.colour_embeddings = self.find_colour_embeddings()

    def find_colour_embeddings(self):
        red_green_line = lambda x: -x + 255
        ordered_sentiments = sorted([int(s) for s in self.sentiments.labels()])
        scaler = 255.0 / (len(self.sentiments) - 1)
        #                 RED                      GREEN    BLUE
        return {pair[0]: (red_green_line(pair[1]), pair[1], 0) for pair in [(str(sentiment), i * scaler) for i, sentiment in enumerate(ordered_sentiments)]}

    def sentiment_colour(self, sentiment):
        if not self.is_setup():
            return "none"

        embedding = self.colour_embeddings[sentiment]
        return "rgb(%d, %d, %d)" % embedding

    def compute_point_abstractions(self, distance, points):
        reductions = self.dimensionality_reduce(points, NeuralNetwork.HIDDEN_REDUCTION)

        if not self.is_setup():
            colours = {part: {layer: "none" for layer in range(NeuralNetwork.LAYERS)} for part in NeuralNetwork.LSTM_PARTS}
            predictions = {part: {layer: None for layer in range(NeuralNetwork.LAYERS)} for part in NeuralNetwork.LSTM_PARTS}
            return reductions, colours, predictions

        predictions = self.predict_distributions(distance, points)
        colours = self.fit_colours(points, predictions)
        return reductions, colours, predictions

    def dimensionality_reduce(self, points, reduction_size):
        reductions = {}

        for part, subd in points.items():
            reductions[part] = {}

            for layer, point in subd.items():
                bucket_size = math.ceil(len(point) / reduction_size)
                reduction = []
                offset = 0

                while offset < len(point):
                    bucket = point[offset:offset + bucket_size]
                    average = sum(bucket) / float(len(bucket))
                    reduction += [average]
                    offset += bucket_size

                reductions[part][layer] = reduction

        return reductions

    def predict_distributions(self, distance, points):
        xs = adjutant.flat_map([[(part, layer, point) for layer, point in subd.items()] for part, subd in points.items()])
        #xs = adjutant.flat_map([[(part, layer, distance, point) for layer, point in subd.items()] for part, subd in points.items()])
        results = self.predictor.evaluate(xs)
        distribution_predictions = {part: {} for part in NeuralNetwork.LSTM_PARTS}

        for i, x in enumerate(xs):
            part, layer, tmp_point = x
            #part, layer, tmp_distance, tmp_point = x
            ordered_predictions = [item[0] for item in sorted(results[i].distribution.items(), key=lambda item: item[1], reverse=True)]
            distribution_predictions[part][layer] = {prediction: results[i].distribution[prediction] for prediction in ordered_predictions[:NeuralNetwork.TOP_PREDICTIONS]}

        return distribution_predictions

    def fit_colours(self, points, predictions):
        colours = {}
        # Make the maximum target distance (1.0) one quarter the length of a dimension in the colour space.
        scaler = (255.0 / 4)

        for part in NeuralNetwork.LSTM_PARTS:
            colours[part] = {}

            for layer in range(NeuralNetwork.LAYERS):
                invert_distribution = {k: 1.0 - v for k, v in predictions[part][layer].items()}
                prediction_distances = [(k, v * scaler) for k, v in mlbase.softmax(invert_distribution).items()]
                fit, _ = geometry.fit_point([self.colour_embeddings[item[0]] for item in prediction_distances], [item[1] for item in prediction_distances], epsilon=0.1, visualize=False)
                colours[part][layer] = "rgb(%d, %d, %d)" % tuple([round(i) for i in fit])

        return colours

    def weights(self, sequence, distance):
        stepwise_lstm = self.lstm.stepwise(True)

        for x in sequence:
            x_word = self.words.decode(self.words.encode(x, True))
            result, instruments = stepwise_lstm.step(x, NeuralNetwork.INSTRUMENTS)

        points = {}

        for part in NeuralNetwork.LSTM_PARTS:
            points[part] = {}

            for layer in range(NeuralNetwork.LAYERS):
                points[part][layer] = instruments[part][layer]

        point_reductions, point_colours, point_predictions = self.compute_point_abstractions(distance, points)
        embedding_reduction = self.dimensionality_reduce({"a": {"b": instruments["embedding"]}}, NeuralNetwork.EMBEDDING_REDUCTION)["a"]["b"]
        embedding = HiddenState(embedding_reduction, colour="none")
        units = []

        for layer in range(NeuralNetwork.LAYERS):
            states = self.make_hidden_states(layer, point_reductions, point_colours, point_predictions)
            units += [Unit(**states)]

        softmax = LabelDistribution(result.distribution, result.encoding, NeuralNetwork.OUTPUT_WIDTH, lambda word: self.sentiment_colour(word))
        return Layer(embedding, units, softmax, len(sequence) - 1, x_word, result.prediction)

    def make_hidden_states(self, layer, point_reductions, point_colours, point_predictions):
        states = {}

        for part in NeuralNetwork.LSTM_PARTS:
            if part.endswith("_gates"):
                min_max = (0, 1)
            else:
                min_max = (None, None)

            prediction = point_predictions[part][layer]

            if prediction is None:
                label_distribution = None
            else:
                label_distribution = LabelDistribution(prediction, {k: None for k, v in prediction.items()}, colour_fn=lambda word: self.sentiment_colour(word))

            hidden_state = HiddenState(point_reductions[part][layer], min_max, point_colours[part][layer], label_distribution)
            states[NeuralNetwork.SINGULAR[part]] = hidden_state

        return states

    def weight_explain(self, sequence, name, column):
        weights = self.weights(sequence)

        if len(sequence) > 1:
            weights_previous = self.weights(sequence[:-1])
            embedding_previous = weights_previous.embedding
            outputs_previous = [unit.output for unit in weights_previous.units]
        else:
            zeros = HiddenState([0] * NeuralNetwork.HIDDEN_WIDTH)
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
                right_feed: explain[NeuralNetwork.HIDDEN_WIDTH:] if len(explain) > NeuralNetwork.HIDDEN_WIDTH else explain
            }

            if left_feed is not None:
                vectors[left_feed] = explain[:NeuralNetwork.HIDDEN_WIDTH]

            return WeightExplain(vectors, bias[column])
        elif name == "cell":
            left = [0] * NeuralNetwork.HIDDEN_WIDTH
            left[column] = weights.units[layer].forget.vector[column]
            left_feed = "forget_hat-%d" % layer
            right = [0] * NeuralNetwork.HIDDEN_WIDTH
            right[column] = weights.units[layer].remember.vector[column]
            right_feed = "remember_hat-%d" % layer
            return WeightExplain({left_feed: left, right_feed: right}, 0)
        elif name == "forget":
            explain = [0] * NeuralNetwork.HIDDEN_WIDTH
            explain[column] = weights.units[layer].cell_previous.vector[column] * weights.units[layer].forget_gate.vector[column]
            return WeightExplain({"cell_previous-%d" % layer: explain}, 0)
        elif name == "remember":
            explain = [0] * NeuralNetwork.HIDDEN_WIDTH
            explain[column] = weights.units[layer].input_hat.vector[column] * weights.units[layer].remember_gate.vector[column]
            return WeightExplain({"input_hat-%d" % layer: explain}, 0)
        elif name == "output":
            explain = [0] * NeuralNetwork.HIDDEN_WIDTH
            explain[column] = weights.units[layer].cell_hat.vector[column] * weights.units[layer].output_gate.vector[column]
            return WeightExplain({"cell_hat-%d" % layer: explain}, 0)
        elif name == "remember_hat":
            explain = [0] * NeuralNetwork.HIDDEN_WIDTH
            explain[column] = weights.units[layer].remember.vector[column]
            return WeightExplain({"remember-%d" % layer: explain}, 0)
        elif name == "forget_hat":
            explain = [0] * NeuralNetwork.HIDDEN_WIDTH
            explain[column] = weights.units[layer].forget.vector[column]
            return WeightExplain({"forget-%d" % layer: explain}, 0)
        elif name == "cell_hat":
            explain = [0] * NeuralNetwork.HIDDEN_WIDTH
            explain[column] = weights.units[layer].cell.vector[column]
            return WeightExplain({"cell-%d" % layer: explain}, 0)
        elif name == "cell_previous":
            explain = [0] * NeuralNetwork.HIDDEN_WIDTH
            explain[column] = weights.units[layer].cell_previous.vector[column]
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

