
import itertools
import json
import logging
import math
from nltk.tokenize import word_tokenize
import numpy as np
import os
import pdb
import random
from sklearn.manifold import TSNE
import threading

from ml import base as mlbase
from ml import nlp
from ml import nn as ffnn
from nnwd import geometry
from nnwd.models import Layer, Unit, WeightExplain, HiddenState, LabelDistribution
from nnwd import pickler
from nnwd import rnn
from pytils.log import setup_logging, user_log
from pytils import adjutant, check


RESUME_DIR = ".resume"


def create(reviews_stream, epochs, verbose):
    xys = []
    xys_file = os.path.join(RESUME_DIR, "xys.pickle")

    if os.path.exists(xys_file):
        xys = pickler.load(xys_file)
    else:
        for review in reviews_stream:
            text = [word.lower() for word in word_tokenize(review["text"])]

            if len(text) <= 25:
                stars = int(review["stars"])
                assert stars == review["stars"], "%s != %s" % (stars, review["stars"])

                if stars != 3:
                    if stars == 2:
                        stars = 1
                    elif stars == 4:
                        stars = 5

                    stars = str(stars)
                    xys.append((text, stars))

                    if len(xys) > 20000:
                        break

        pickler.dump(xys, xys_file)

    random.shuffle(xys)
    split_1 = int(len(xys) * 0.8)
    split_2 = split_1 + int(len(xys) * 0.1)
    train_xys = xys[:split_1]
    validation_xys = xys[split_1:split_2]
    test_xys = xys[split_2:]
    logging.debug("data sets (train, validation, test): %d, %d, %d" % (len(train_xys), len(validation_xys), len(test_xys)))

    for data_set in [train_xys, validation_xys, test_xys]:
        histogram = {}

        for xy in data_set:
            if xy[1] not in histogram:
                histogram[xy[1]] = 0

            histogram[xy[1]] += 1

        logging.debug("data set histogram: %s" % histogram)

    vocabulary = set([mlbase.BLANK])
    classes = set()

    for xy in xys:
        for word in xy[0]:
            vocabulary.add(word)

        classes.add(xy[1])

    words = mlbase.Labels(vocabulary, unknown=nlp.UNKNOWN)
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
        self.train_xys = [mlbase.Xy(*pair) for pair in train_xys]
        self.validation_xys = [mlbase.Xy(*pair) for pair in validation_xys]
        self.test_xys = [mlbase.Xy(*pair) for pair in test_xys]
        self.epoch_threshold = epoch_threshold
        self.setup_complete = False
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
        self.lstm = rnn.Rnn(NeuralNetwork.LAYERS, NeuralNetwork.HIDDEN_WIDTH, NeuralNetwork.EMBEDDING_WIDTH, self.words, self.sentiments)
        lstm_dir = os.path.join(RESUME_DIR, "lstm")

        if os.path.exists(lstm_dir):
            self.lstm.load(lstm_dir)
        else:
            accuracy_validation = 0.0
            batch = 512
            arc = -1

            while accuracy_validation < 0.8 and arc < 12:
                arc += 1

                if arc % 2 == 0:
                    batch = max(8, int(batch / 2))

                training_parameters = mlbase.TrainingParameters() \
                    .batch(batch) \
                    .epochs(self.epoch_threshold) \
                    .convergence(False)
                loss = self.lstm.train(self.train_xys, training_parameters)
                accuracy_validation = self.lstm.test(self.validation_xys)
                logging.debug("train lstm arc %d (batch %d): (loss, accuracy) (%s, %s)" % (arc, batch, loss, accuracy_validation))

            self.lstm.save(lstm_dir)

        logging.debug("Calculating final validation accuracy.")
        accuracy_validation = self.lstm.test(self.validation_xys, True)
        logging.debug("Calculating final test accuracy.")
        accuracy_test = self.lstm.test(self.test_xys, True)
        logging.debug("(v, t): (%s, %s)" % (accuracy_validation, accuracy_test))

    def _train_predictor(self):
        part_labels = mlbase.Labels(set(NeuralNetwork.INSTRUMENTS))
        layer_labels = mlbase.Labels(set(range(NeuralNetwork.LAYERS)))
        distance_field = mlbase.IntegerField()
        hidden_vector = mlbase.VectorField(NeuralNetwork.HIDDEN_WIDTH)
        #predictor_input = mlbase.ConcatField([part_labels, layer_labels, hidden_vector])
        predictor_input = mlbase.ConcatField([part_labels, layer_labels, distance_field, hidden_vector])
        predictor_output = mlbase.Labels(set(self.sentiments.labels()))
        hyper_parameters = ffnn.HyperParameters() \
            .width(len(predictor_input))
        self.predictor = ffnn.Model("predictor", hyper_parameters, predictor_input, predictor_output, mlbase.SINGLE_LABEL)
        self.predictor_xys = []
        predictor_xys_file = os.path.join(RESUME_DIR, "predictor_xys.pickle")

        if os.path.exists(predictor_xys_file):
            self.predictor_xys = pickler.load(predictor_xys_file)
        else:
            for xy in self.train_xys:
                stepwise_lstm = self.lstm.stepwise(False)
                xs = []
                final_distribution = None

                for i, word in enumerate(xy.x):
                    result, instruments = stepwise_lstm.step(word, NeuralNetwork.INSTRUMENTS)
                    distance = len(xy.x) - i - 1

                    if distance == 0:
                        final_distribution = result.distribution

                    #x = ("embedding", 0, tuple(instruments["embedding"]))
                    x = ("embedding", 0, distance, tuple(instruments["embedding"]))
                    xs += [x]
                    #train_xy = mlbase.Xy(x, result.distribution)
                    #self.predictor_xys += [train_xy]

                    for part in NeuralNetwork.LSTM_PARTS:
                        for layer in range(NeuralNetwork.LAYERS):
                            point = tuple(instruments[part][layer])
                            #x = (part, layer, point)
                            x = (part, layer, distance, point)
                            xs += [x]
                            #train_xy = mlbase.Xy(x, result.distribution)
                            #self.predictor_xys += [train_xy]

                for x in xs:
                    train_xy = mlbase.Xy(x, final_distribution)
                    self.predictor_xys += [train_xy]

            pickler.dump(self.predictor_xys, predictor_xys_file)

        logging.debug("Predictor data (distance based): %d." % len(self.predictor_xys))
        self.setup_complete = True
        training_parameters = mlbase.TrainingParameters() \
            .epochs(self.epoch_threshold) \
            .batch(32)
        loss = self.predictor.train(self.predictor_xys, training_parameters)
        logging.debug("train predictor %s" % loss)

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
            return None

        embedding = self.colour_embeddings[sentiment]
        return "rgb(%d, %d, %d)" % embedding

    def compute_point_abstractions(self, distance, points):
        reductions = self.dimensionality_reduce(points, NeuralNetwork.HIDDEN_REDUCTION)

        if not self.is_setup():
            colours = {key: None for key, point in points.items()}
            predictions = {key: None for key, point in points.items()}
            return reductions, colours, predictions

        predictions = self.predict_distributions(distance, points)
        colours = self.fit_colours(points, predictions)
        return reductions, colours, predictions

    def dimensionality_reduce(self, points, reduction_size):
        reductions = {}

        for key, point in points.items():
            bucket_size = math.ceil(len(point) / reduction_size)
            reduction = []
            offset = 0

            while offset < len(point):
                bucket = point[offset:offset + bucket_size]
                average = sum(bucket) / float(len(bucket))
                reduction += [average]
                offset += bucket_size

            reductions[key] = reduction

        return reductions

    def predict_distributions(self, distance, points):
        #xs = [self.decode_key(key) + (point,) for key, point in points.items()]
        xs = [self.decode_key(key) + (distance, point) for key, point in points.items()]
        results = self.predictor.evaluate(xs)
        distribution_predictions = {}

        for i, x in enumerate(xs):
            #part, layer, _ = x
            part, layer, distance, _ = x
            ordered_predictions = [item[0] for item in sorted(results[i].distribution.items(), key=lambda item: item[1], reverse=True)]
            distribution_predictions[self.encode_key(part, layer)] = {prediction: results[i].distribution[prediction] for prediction in ordered_predictions[:NeuralNetwork.TOP_PREDICTIONS]}

        return distribution_predictions

    def fit_colours(self, points, predictions):
        colours = {}
        # Make the maximum target distance (1.0) one quarter the length of a dimension in the colour space.
        scaler = (255.0 / 4)

        for key, point in points.items():
            invert_distribution = {k: 1.0 - v for k, v in predictions[key].items()}
            prediction_distances = [(k, v * scaler) for k, v in mlbase.softmax(invert_distribution).items()]
            fit, _ = geometry.fit_point([self.colour_embeddings[item[0]] for item in prediction_distances], [item[1] for item in prediction_distances], epsilon=0.1, visualize=False)
            colours[key] = "rgb(%d, %d, %d)" % tuple([round(i) for i in fit])

        return colours

    def weights(self, sequence, distance):
        stepwise_lstm = self.lstm.stepwise(handle_unknown=True)

        for x in sequence:
            x_word = self.words.decode(self.words.encode(x, True))
            result, instruments = stepwise_lstm.step(x, NeuralNetwork.INSTRUMENTS)

        embedding_key = self.encode_key("embedding")
        points = {
            embedding_key: instruments["embedding"]
        }

        for part in NeuralNetwork.LSTM_PARTS:
            for layer in range(NeuralNetwork.LAYERS):
                points[self.encode_key(part, layer)] = instruments[part][layer]

        point_reductions, point_colours, point_predictions = self.compute_point_abstractions(distance, points)
        embedding = HiddenState(point_reductions[embedding_key], colour=point_colours[embedding_key], predictions=self.prediction_distribution(point_predictions[embedding_key]))
        units = []

        for layer in range(NeuralNetwork.LAYERS):
            states = self.make_hidden_states(layer, point_reductions, point_colours, point_predictions)
            units += [Unit(**states)]

        softmax = LabelDistribution(result.distribution, NeuralNetwork.OUTPUT_WIDTH, lambda sentiment: self.sentiment_colour(sentiment))
        return Layer(embedding, units, softmax, len(sequence) - 1, x_word, result.prediction)

    def encode_key(self, part, layer=0):
        return "%s-%d" % (part, layer)

    def decode_key(self, key):
        result = key.split("-")

        if len(result) == 2:
            return (result[0], int(result[1]))
        else:
            raise ValueError("invalid decode of '%s': %s" % (key, result))

    def prediction_distribution(self, prediction):
        if prediction is None:
            return None

        return LabelDistribution(prediction, colour_fn=lambda sentiment: self.sentiment_colour(sentiment))

    def make_hidden_states(self, layer, point_reductions, point_colours, point_predictions):
        states = {}

        for part in NeuralNetwork.LSTM_PARTS:
            if part.endswith("_gates"):
                min_max = (0, 1)
            else:
                min_max = (None, None)

            key = self.encode_key(part, layer)
            hidden_state = HiddenState(point_reductions[key], min_max, point_colours[key], self.prediction_distribution(point_predictions[key]))
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

