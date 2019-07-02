
import collections
from csv import writer as csv_writer
import functools
import itertools
import json
import logging
import math
from nltk.tokenize import word_tokenize
import numpy as np
import os
import pickle
import pdb
import queue
import random
import resource
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import sqlite3
import string
import sys
import threading
import time
import uuid

from ml import base as mlbase
from ml import model
from ml import nlp
from nnwd import data
from nnwd import geometry
from nnwd import latex
from nnwd import lm
from nnwd.models import Timestep, WeightExplain, WeightDetail, HiddenState, LabelDistribution, SequenceRollup, SequenceMatch, Estimate, SoftFilters, Predicates
from nnwd import parameters
from nnwd import pickler
from nnwd import query
from nnwd import reduction
from nnwd import rnn
from nnwd import semantic
from nnwd import sequential
from nnwd import states
from pytils.log import user_log
from pytils import adjutant, check


def sa_colour_mapping():
    # Pallet from: http://nlp.stanford.edu/sentiment/treebank.html
    #  very negative: 103, 0, 31
    #       negative: 214, 96, 77
    #        neutral: 247, 247, 247
    #       positive: 67, 147, 195
    #  very positive: 5, 48, 97
    return {
        "very negative": (214, 96, 77),
        "negative": (214, 96, 77),
        "neutral": (247, 247, 247),
        "positive": (67, 147, 195),
        "very positive": (67, 147, 195),
    }


def parens_colour_mapping():
    # Pallet from: http://colorbrewer2.org/#type=qualitative&scheme=Accent&n=4
    #  0: 127,201,127
    #  1: 190,174,212
    #  2: 253,192,134
    #  3: 255,255,153
    #  4: 56,108,176
    green = (126, 201, 126)
    purple = (189, 174, 210)
    orange = (234, 222, 123)
    terminal = (255, 255, 153)
    other = (56, 108, 176)
    return {
        "(": green,
        ")": purple,
        "0": orange,
        "1": orange,
        "2": orange,
        "3": orange,
        "4": orange,
        ".": terminal,
        mlbase.BLANK: other,
        nlp.UNKNOWN: other,
    }


def coarse_colour_mapping():
    green = (127, 201, 127)
    purple = (190, 174, 212)
    orange = (253, 192, 134)
    yellow = (255, 255, 153)
    blue = (56, 108, 176)
    return {
        "NOUN": green,
        "VERB": purple,
        "ADV": orange,
        "ADJ": blue,
        "OTHER": yellow,
    }


def pos_colour_mapping():
    blue = (51, 102, 204)
    red = (220, 57, 18)
    orange = (255, 153, 0)
    green = (16, 150, 24)
    purple = (153, 0, 153)
    light_blue = (0, 153, 198)
    pink = (221, 68, 119)
    light_green = (102, 170, 0)
    dark_red = (184, 46, 46)
    dark_blue = (49, 99, 149)
    yellow = (170, 170, 17)
    return {
        "CC": purple,
        "CD": yellow,
        "DT": blue,
        "EX": dark_blue,
        "FW": yellow,
        "IN": purple,
        "JJ": light_blue,
        "JJR": light_blue,
        "JJS": light_blue,
        "LS": pink,
        "MD": red,
        "NN": red,
        "NNS": red,
        "NNP": red,
        "NNPS": red,
        "PDT": dark_blue,
        "POS": dark_blue,
        "PRP": orange,
        "PRP$": orange,
        "RB": light_green,
        "RBR": light_green,
        "RBS": light_green,
        "RP": dark_red,
        "SYM": pink,
        "TO": dark_blue,
        "UH": dark_blue,
        "VB": green,
        "VBD": green,
        "VBG": green,
        "VBN": green,
        "VBP": green,
        "VBZ": green,
        "WDT": blue,
        "WP": orange,
        "WP$": orange,
        "WRB": light_green,
        "PUNCT": pink,
    }


class NeuralNetwork:
    def __init__(self, data_dir, sequential_dir, buckets_dir, encoding_dir, use_fixed_buckets):
        self.data_dir = data_dir
        self.sequential_dir = sequential_dir
        self.buckets_dir = buckets_dir
        self.encoding_dir = encoding_dir
        self.words = data.get_words(self.data_dir)
        description = data.get_description(self.data_dir)

        if description.task == data.LM:
            self.outputs = self.words
            # Map output words to their POS tags.
            pos_mapping = data.get_pos_mapping(self.data_dir)
            #self.output_mapping = lambda output: pos_mapping[output] if output in pos_mapping else "NN"
            #self.colour_mapping = pos_colour_mapping()
            self.output_mapping = lambda output: lm.COARSE_MAP[pos_mapping[output] if output in pos_mapping else "NN"]
            self.colour_mapping = coarse_colour_mapping()
            self.sort_key = lambda key_value: -key_value[1]
        else:
            self.outputs = data.get_outputs(self.data_dir)
            self.output_mapping = lambda output: output
            self.colour_mapping = sa_colour_mapping()
            self.sort_key = lambda key_value: sa.sentiment_sort_key(key_value[1])

        self.top_k = max(1, int(len(self.outputs) * parameters.SEM_TOP_K_PERCENT))

        if use_fixed_buckets:
            self.bucket_mappings = reduction.get_fixed_buckets(self.buckets_dir)
        else:
            self.bucket_mappings = reduction.get_learned_buckets(self.buckets_dir)

        self.lstm = sequential.load_model(self.data_dir, self.sequential_dir)

        def _ffnn_constructor(scope, hyper_parameters, extra, case_field, hidden_vector, word_labels, output_labels):
            if extra["word_input"]:
                input_field = mlbase.ConcatField([case_field, hidden_vector, word_labels])
            else:
                input_field = mlbase.ConcatField([case_field, hidden_vector])

            if extra["monolith"]:
                return model.Ffnn(scope, hyper_parameters, extra, input_field, output_labels)
            else:
                return model.SeparateFfnn(scope, hyper_parameters, extra, input_field, output_labels, case_field)

        self.sem = semantic.load_model(self.lstm, self.encoding_dir, model_fn=_ffnn_constructor)

        # TODO
        embedding_padding = tuple([0] * max(0, self.lstm.hyper_parameters.width - self.lstm.hyper_parameters.embedding_width))
        hidden_padding = tuple([0] * max(0, self.lstm.hyper_parameters.embedding_width - self.lstm.hyper_parameters.width))

        #if hasattr(model, "extra") and model.extra["word_input"]:
        #    def converter(key, hidden_state):
        #        return (key, tuple(hidden_state.point) + (embedding_padding if self.lstm.is_embedding(key) else hidden_padding), hidden_state.word)
        #else:
        def _as_input(key, point):
            return (key, tuple(point) + (embedding_padding if self.lstm.is_embedding(key) else hidden_padding))

        self.as_input = _as_input
        self.details_mins = {}
        self.details_maxs = {}
        self.weights_mins = {}
        self.weights_maxs = {}

    @functools.lru_cache()
    def stepwise(self, sequence):
        if len(sequence) == 0:
            return rnn.Stepwise(self.lstm, "root", handle_unknown=True)
        else:
            return self.stepwise(sequence[:-1]).next_stepwise(sequence[-1])

    def query_lstm(self, sequence, instruments):
        word = sequence[-1]
        resolved_word = self.words.decode(self.words.encode(word, True))
        stepwise_lstm = self.stepwise(tuple(sequence[:-1]))
        assert stepwise_lstm.name == ",".join(["root"] + sequence[:-1]), "%s != %s" % (stepwise_lstm.name, ",".join(["root"] + sequence))
        result, instrument_values = stepwise_lstm.query(word, instrument_names=instruments)

        for kind, units in instrument_values.items():
            if kind != "ws":
                for unit, vector in enumerate(units):
                    key = self.lstm.encode_key(kind, unit)
                    minimum = min(vector)

                    if key not in self.details_mins or minimum < self.details_mins[key]:
                        self.details_mins[key] = minimum

                    maximum = max(vector)

                    if key not in self.details_maxs or maximum > self.details_maxs[key]:
                        self.details_maxs[key] = maximum

        return resolved_word, result, instrument_values

    def _dummy(self):
        weights = []
        order = []

        for word in self.words.labels():
            weight = self.lstm.embed(word)
            weights += [weight]
            order += [word]

        # Pallet from: http://colorbrewer2.org/#type=qualitative&scheme=Accent&n=6
        #  0: 127,201,127
        #  1: 190,174,212
        #  2: 253,192,134
        #  3: 255,255,153
        #  4: 56,108,176
        #  5: 240,2,127
        pallet = {
              0: (127, 201, 127),
              1: (190, 174, 212),
              2: (253, 192, 134),
              3: (255, 255, 153),
              4: (56, 108, 176),
              5: (240, 2, 127),
        }
        self.colour_mapping = self.find_colour_embeddings(weights, order, pallet)

    def find_colour_embeddings(self, weights, order, pallet):
        colour_embeddings = {}

        for tsne_perplexity in [50]:#[5, 10, 20, 30, 50]:
            tsne = TSNE(n_components=1, perplexity=tsne_perplexity)
            embeddings_1d = adjutant.flat_map(tsne.fit_transform(weights))
            minimum = min(embeddings_1d)
            maximum = max(embeddings_1d)
            domain_step = (maximum - minimum) / len(pallet)
            category = lambda point: len(pallet) - 1 if point == maximum else int((point - minimum) / domain_step)
            colour_embeddings[tsne_perplexity] = {order[j]: pallet[category(embedding)] for j, embedding in enumerate(embeddings_1d)}
            #mapping = {}

            #for word, colour_point in colour_embeddings[tsne_perplexity].items():
            #    if colour_point not in mapping:
            #        mapping[colour_point] = []

            #    mapping[colour_point] += [word]

            #logging.debug(adjutant.dict_as_str(mapping))

        #choice = input("choice (05, 10, 20, 30, 50): ")
        return colour_embeddings[50]

    def mapped_output(self, output):
        resolved = self.outputs.decode(self.outputs.encode(output, True))
        return self.output_mapping(resolved)

    def output_colour(self, output):
        return self.colour_mapping[self.mapped_output(output)]

    def rgb(self, colour):
        return "rgb(%d, %d, %d)" % colour

    def compute_point_abstractions(self, word, points):
        reductions = self.dimensionality_reduce(points)
        predictions = self.predict_distributions(word, points)
        colours = self.fit_colours(points, predictions)
        return reductions, colours, predictions

    def dimensionality_reduce(self, points):
        out = {key: reduction.reduce(self.bucket_mappings[key], point) for key, point in points.items()}

        for key, vector in out.items():
            minimum = min(vector)

            if key not in self.weights_mins or minimum < self.weights_mins[key]:
                self.weights_mins[key] = minimum

            maximum = max(vector)

            if key not in self.weights_maxs or maximum > self.weights_maxs[key]:
                self.weights_maxs[key] = maximum

        return out

    def predict_distributions(self, word, points):
        annotation = None
        keys = []
        xys = []

        for key, point in points.items():
            keys += [key]
            xys += [mlbase.Xy(self.as_input(key, point), annotation)]

        results, _ = self.sem.evaluate(xys)

        distribution_predictions = {}

        for i, key in enumerate(keys):
            ordered_predictions = [item[0] for item in sorted(results[i].distribution().items(), key=lambda item: item[1], reverse=True)]
            distribution_predictions[key] = {prediction: results[i].distribution()[prediction] for prediction in ordered_predictions[:self.top_k]}

        return distribution_predictions

    def fit_colours(self, points, predictions):
        colours = {}

        for key, point in points.items():
            #colours[key] = self._fit_averaging(predictions[key])
            #colours[key] = self._fit_top_k(predictions[key])
            colours[key] = self._fit_top_2_special(predictions[key])

        return colours

    def _fit_averaging(self, predictions):
        ### Averaging - which isn't going to be right.
        average = [0, 0, 0]

        for output, probability in predictions.items():
            self.mapped_output(output)
            colour = self.output_colour(output)
            average = [average[i] + (colour[i]) for i in range(3)]

        return "rgb(%d, %d, %d)" % tuple([round(i / len(predictions)) for i in average])

    def _fit_top_k(self, predictions):
        ## Fitting across top-k points.
        colour_probabilities = {}

        for output, probability in predictions.items():
            self.mapped_output(output)
            colour = self.output_colour(output)

            if colour not in colour_probabilities:
                colour_probabilities[colour] = 0

            colour_probabilities[colour] += probability

        maximum_distance = None

        for pair in itertools.combinations([colour for colour in colour_probabilities.keys()], 2):
            distance = geometry.distance(pair[0], pair[1])

            if maximum_distance is None or distance > maximum_distance:
                maximum_distance = distance

        inverted = [item for item in mlbase.regmax({c: 1.0 - p for c, p in colour_probabilities.items()}).items()]
        fit, _ = geometry.fit_point([item[0] for item in inverted], [item[1] * maximum_distance for item in inverted])
        return "rgb(%d, %d, %d)" % tuple([round(i) for i in fit])

    def _fit_top_2_special(self, predictions):
        ### Fitting across top-2 points (done differently than top-k, k=2).
        output_colourings = {}

        for output, probability in predictions.items():
            self.mapped_output(output)
            colour = self.output_colour(output)
            output_colourings[output] = (colour, probability)

        interpolation_points = {}

        for output, colour_probability in sorted(output_colourings.items(), key=lambda item: item[1][1], reverse=True)[:2]:
            colour, probability = colour_probability
            interpolation_points[colour] = (output, probability)

        if len(interpolation_points) == 1:
            return "rgb(%d, %d, %d)" % next(iter(interpolation_points.keys()))
        else:
            point_a, point_b = [item for item in interpolation_points.items()]
            distance = geometry.distance(point_a[0], point_b[0])
            # Not a typo: we want to invert their probabilities so that the most likely prediction gets the smallest distance, and visa versa.
            #                            v              v              v              v
            pdist = mlbase.regmax({point_a[1][0]: point_b[1][1], point_b[1][0]: point_a[1][1]})
            fit = geometry.fit_proportion((point_a[0], point_b[0]), (pdist[point_a[1][0]], pdist[point_b[1][0]]))
            return "rgb(%d, %d, %d)" % tuple([round(i) for i in fit])

    def weights(self, sequence):
        last_word, result, instrument_values = self.query_lstm(sequence, rnn.LSTM_INSTRUMENTS)
        points = {}

        for part, layer in self.lstm.part_layers():
            points[self.lstm.encode_key(part, layer)] = instrument_values[part][layer]

            if self.lstm.is_embedding(part, layer):
                embedding_key = self.lstm.encode_key(part, layer)

        point_reductions, point_colours, point_predictions = self.compute_point_abstractions(last_word, points)
        embedding_name = self.latex_name(len(sequence) - 1, "embedding")
        embedding_name_no_t = self.latex_name_no_t("embedding")
        min_max = (self.weights_mins[embedding_key], self.weights_maxs[embedding_key])
        embedding = HiddenState(embedding_name, embedding_name_no_t, point_reductions[embedding_key], min_max, point_colours[embedding_key], self.prediction_distribution(point_predictions[embedding_key]))
        units = self.make_lstm_units(len(sequence) - 1, point_reductions, point_colours, point_predictions)
        softmax_name = self.latex_name(len(sequence) - 1, "softmax")
        softmax = LabelDistribution(softmax_name, result.distribution, self.sort_key, parameters.OUTPUT_TOP_K, lambda output: self.rgb(self.output_colour(output)))
        return Timestep(embedding, units, softmax, len(sequence) - 1, last_word, result.prediction)

    def weight_detail(self, sequence, part, layer):
        last_word, result, instrument_values = self.query_lstm(sequence, rnn.LSTM_INSTRUMENTS)
        point = instrument_values[part][layer]
        # This is the regular process by which a hidden_state is served out with its various reductions/abstractions.
        key = self.lstm.encode_key(part, layer)
        keyed_point = {key: point}
        reduction, colour, prediction = self.compute_point_abstractions(last_word, keyed_point)
        name = self.latex_name(len(sequence) - 1, part, layer)
        name_no_t = self.latex_name_no_t(part, layer)
        min_max = (self.details_mins[key], self.details_maxs[key])
        hidden_state = HiddenState(name, name_no_t, reduction[key], min_max, colour[key], self.prediction_distribution(prediction[key]))
        back_links = {}
        repositioned_point = [[] for i in range(len(point))]
        positioning = {}
        i = 0

        for bucket, dimensions in self.bucket_mappings[key].items():
            for dimension in dimensions:
                assert i not in back_links, "%d already in %s" % (i, back_links)
                back_links[i] = bucket
                repositioned_point[i] = point[dimension]
                positioning[i] = dimension
                i += 1

        # This is the full point.
        full_hidden_state = HiddenState(name, name_no_t, repositioned_point, min_max, None, None, positioning)
        return WeightDetail(hidden_state, full_hidden_state, back_links)

    def soft_filter(self, sequence):
        last_word, result, instrument_values = self.query_lstm(sequence, ["ws"])
        part = "w"
        ws = []

        for timestep in range(len(instrument_values["ws"])):
            units = []

            for layer in range(len(instrument_values["ws"][timestep])):
                point = instrument_values["ws"][timestep][layer]
                # TODO
                point_reduction = reduction.reduce(self.bucket_mappings["cells-0"], point)
                name = self.latex_name(timestep, part, layer)
                name_no_t = self.latex_name_no_t(part, layer)
                hidden_state = HiddenState(name, name_no_t, point_reduction, (0, 1))
                units += [hidden_state]

            ws += [units]

        return last_word, ws

    def soft_filters(self, sequence):
        word_timesteps = [self.soft_filter(sequence[:i + 1]) for i in range(len(sequence))]
        return SoftFilters([item[0] for item in word_timesteps], [item[1] for item in word_timesteps])

    def prediction_distribution(self, prediction):
        if prediction is None:
            return None

        return LabelDistribution(None, prediction, self.sort_key, parameters.PSE_TOP_K, colour_fn=lambda output: self.rgb(self.output_colour(output)))

    def make_lstm_units(self, timestep, point_reductions, point_colours, point_predictions):
        units = {}

        # TODO
        for part in rnn.LSTM_PARTS:
            units[part] = {}

            for layer in range(self.lstm.hyper_parameters.layers):
                key = self.lstm.encode_key(part, layer)
                name = self.latex_name(timestep, part, layer)
                name_no_t = self.latex_name_no_t(part, layer)
                min_max = (self.weights_mins[key], self.weights_maxs[key])
                hidden_state = HiddenState(name, name_no_t, point_reductions[key], min_max, point_colours[key], self.prediction_distribution(point_predictions[key]))
                units[part][layer] = hidden_state

        return units

    def weight_explain(self, sequence, name, column):
        weights = self.weights(sequence)

        if len(sequence) > 1:
            weights_previous = self.weights(sequence[:-1])
            embedding_previous = weights_previous.embedding
            outputs_previous = [unit.output for unit in weights_previous.units]
        else:
            zeros = HiddenState([0] * parameters.HIDDEN_WIDTH)
            embedding_previous = zeros
            outputs_previous = [zeros] * parameters.LAYERS

        parts = name.split("-")
        name = parts[0]

        if len(parts) == 2:
            layer = int(parts[1])
        else:
            layer = None

        if name in parameters.MATRICES:
            matrix = self.lstm.probe(parameters.MATRICES[name], layer)
            bias = self.lstm.probe(parameters.MATRICES[name] + "_bias", layer)

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
                u = layer - 1 if layer is not None else parameters.LAYERS - 1
                right = weights.units[u].output.vector
                right_feed = "output-%d" % (u)

            explain = ((left + right) * matrix[:, column])
            vectors = {
                right_feed: explain[parameters.HIDDEN_WIDTH:] if len(explain) > parameters.HIDDEN_WIDTH else explain
            }

            if left_feed is not None:
                vectors[left_feed] = explain[:parameters.HIDDEN_WIDTH]

            return WeightExplain(vectors, bias[column])
        elif name == "cell":
            left = [0] * parameters.HIDDEN_WIDTH
            left[column] = weights.units[layer].forget.vector[column]
            left_feed = "forget_hat-%d" % layer
            right = [0] * parameters.HIDDEN_WIDTH
            right[column] = weights.units[layer].remember.vector[column]
            right_feed = "remember_hat-%d" % layer
            return WeightExplain({left_feed: left, right_feed: right}, 0)
        elif name == "forget":
            explain = [0] * parameters.HIDDEN_WIDTH
            explain[column] = weights.units[layer].cell_previous.vector[column] * weights.units[layer].forget_gate.vector[column]
            return WeightExplain({"cell_previous-%d" % layer: explain}, 0)
        elif name == "remember":
            explain = [0] * parameters.HIDDEN_WIDTH
            explain[column] = weights.units[layer].input_hat.vector[column] * weights.units[layer].remember_gate.vector[column]
            return WeightExplain({"input_hat-%d" % layer: explain}, 0)
        elif name == "output":
            explain = [0] * parameters.HIDDEN_WIDTH
            explain[column] = weights.units[layer].cell_hat.vector[column] * weights.units[layer].output_gate.vector[column]
            return WeightExplain({"cell_hat-%d" % layer: explain}, 0)
        elif name == "remember_hat":
            explain = [0] * parameters.HIDDEN_WIDTH
            explain[column] = weights.units[layer].remember.vector[column]
            return WeightExplain({"remember-%d" % layer: explain}, 0)
        elif name == "forget_hat":
            explain = [0] * parameters.HIDDEN_WIDTH
            explain[column] = weights.units[layer].forget.vector[column]
            return WeightExplain({"forget-%d" % layer: explain}, 0)
        elif name == "cell_hat":
            explain = [0] * parameters.HIDDEN_WIDTH
            explain[column] = weights.units[layer].cell.vector[column]
            return WeightExplain({"cell-%d" % layer: explain}, 0)
        elif name == "cell_previous":
            explain = [0] * parameters.HIDDEN_WIDTH
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

    @latex.generate_png
    def latex_name(self, timestep, part, layer=None):
        t = timestep + 1
        u = None if layer is None else layer + 1

        if part == "embedding":
            return "e_%d" % t
        elif part == "remember_gates":
            return "i_%d^%d" % (t, u)
        elif part == "forget_gates":
            return "f_%d^%d" % (t, u)
        elif part == "output_gates":
            return "o_%d^%d" % (t, u)
        elif part == "input_hats":
            return "tilde_c_%d^%d" % (t, u)
        elif part == "remembers":
            return "s_%d^%d" % (t, u)
        elif part == "forgets":
            return "l_%d^%d" % (t, u)
        elif part == "cells":
            return "c_%d^%d" % (t, u)
        elif part == "cell_hats":
            return "cellhat_%d^%d" % (t, u)
        elif part == "cell_previouses":
            return "c_%d^%d" % (t - 1, u)
        elif part == "outputs":
            return "h_%d^%d" % (t, u)
        elif part == "softmax":
            return "y_%d" % t
        elif part == "w":
            return "w_%d^%d" % (t, u)

    @latex.generate_png
    def latex_name_no_t(self, part, layer=None):
        u = None if layer is None else layer + 1

        if part == "embedding":
            return "e"
        elif part == "remember_gates":
            return "i^%d" % u
        elif part == "forget_gates":
            return "f^%d" % u
        elif part == "output_gates":
            return "o^%d" % u
        elif part == "input_hats":
            return "tilde_c^%d" % u
        elif part == "remembers":
            return "s^%d" % u
        elif part == "forgets":
            return "l^%d" % u
        elif part == "cells":
            return "c^%d" % u
        elif part == "cell_hats":
            return "cellhat^%d" % u
        elif part == "cell_previouses":
            return "c^%d" % u
        elif part == "outputs":
            return "h^%d" % u
        elif part == "softmax":
            return "y"
        elif part == "w":
            return "w^%d" % u


class PatternEngine:
    def __init__(self, neural_network):
        self.neural_network = neural_network

    def match(self, tolerance, use_skip_empties, use_consistent_features, annotated_sequences, patterns):
        # input
        # annotated_sequences: list of sequences, where each sequence is a tuple of the word and monotonically increasing pattern index
        #       [
        #           ([1, 2], [ the, little, prince ]),
        #           ...
        #       ]
        # patterns: list of pattern
        #       [ [cell-0, forget-1], ... ]
        #
        # output
        # predicates: list of dicts, keyed by rnn part keys to lists of (axis, value) features
        #       [ {cell-0: [ (0, 0.5), (22, -0.02), ... ] }, ... ]
        targets = [{key: [] for key in pattern} for pattern in patterns]
        pattern_instruments = [set([self.neural_network.lstm.decode_key(key)[0] for key in pattern]) for pattern in patterns]

        for annotated_sequence in annotated_sequences:
            annotations, sequence = annotated_sequence
            assert len(annotations) == len(patterns)

            for index, annotation in enumerate(annotations):
                last_word, result, instrument_values = self.neural_network.query_lstm(sequence[:annotation + 1], pattern_instruments[index])

                for key in patterns[index]:
                    part, layer = self.neural_network.lstm.decode_key(key)
                    targets[index][key] += [instrument_values[part][layer]]

        predicates = []
        consistent_features = {}

        for index, dataset in enumerate(targets):
            predicate = {}

            for key, points in dataset.items():
                features = self._intersecting_features(tolerance, use_skip_empties, points)

                if len(features) > 0:
                    predicate[key] = features
                    axis = set([axis for axis in features.keys()])

                    if key not in consistent_features:
                        consistent_features[key] = axis
                    else:
                        consistent_features[key].intersection_update(axis)

            predicates += [predicate]

        if use_consistent_features:
            updates = []

            for predicate in predicates:
                update = {}

                for key, features in predicate.items():
                    if len(consistent_features[key]) > 0:
                        update_features = {axis: features[axis] for axis in consistent_features[key]}
                        update[key] = update_features

                if len(update) > 0:
                    updates += [update]

            predicates = updates

        return Predicates(predicates)

    def _intersecting_features(self, tolerance, use_skip_empties, dataset):
        tolerance_2 = tolerance * 2
        features = {}

        for axis in range(len(dataset[0])):
            minimum = None
            maximum = None
            match = True

            for point in dataset:
                if minimum is None or point[axis] < minimum:
                    minimum = point[axis]

                if maximum is None or point[axis] > maximum:
                    maximum = point[axis]

                if minimum is not None and maximum is not None:
                    if maximum - minimum > tolerance_2:
                        match = False
                        break

            if match:
                value = minimum + ((maximum - minimum) / 2.0)

                if not use_skip_empties or not math.isclose(value, 0, abs_tol=1e-03):
                    features[axis] = value

        return features


class QueryEngine:
    def __init__(self, neural_network, query_dir, db_kind):
        self.neural_network = neural_network
        self.query_dir = query_dir
        self.db_kind = db_kind
        self.requests = queue.Queue()
        self.responses = {}
        thread = threading.Thread(target=self._process_sql)
        thread.daemon = True
        thread.start()

    def _process_sql(self):
        # Sql requests need to be run in the same thread as where the handle is created - so we do this using IPC.
        query_dbs = query.get_databases(self.query_dir, self.db_kind, self.neural_network.lstm)
        logging.debug("Started sql IPC.")

        while True:
            item = self.requests.get()

            while not self.requests.empty():
                request_id, request = item
                self.responses[request_id] = "shortcircuit"
                item = self.requests.get()

            if item is not None:
                request_id, request = item
                key, axis_target, tolerance, matched_sequences = request

                try:
                    if matched_sequences is None:
                        axis, target = axis_target
                        self.responses[request_id] = query_dbs[key].select_activations_range(axis, target - tolerance, target + tolerance)
                    else:
                        self.responses[request_id] = query_dbs[key].select_activations(matched_sequences)
                except sqlite3.DatabaseError as e:
                    # No idea why this is happening.. just drop and move on.
                    logging.debug("sqlite3 error: %s" % str(e))
                    self.responses[request_id] = None

    def find_estimate(self, tolerance, predicates):
        matches = self.find_matches(tolerance, True, predicates)
        lower = len(matches)
        return Estimate(lower=lower, upper=None if lower > 10 else lower * 10)

    def find(self, tolerance, predicates):
        matches = self.find_matches(tolerance, False, predicates)
        rollups = {}

        with open("found-matches.txt", "w") as fh:
            for sequence, path in matches:
                fh.write("%s|%s\n" % (" ".join(sequence), ",".join([str(i) for i in path])))
                matched_words = []
                elides = []
                last_index = None

                for index in path:
                    word = sequence[index]
                    matched_words += [word]

                    if last_index is None:
                        elides += [index != 0]
                    elif last_index + 1 == index:
                        elides += [False]
                    else:
                        elides += [True]

                    last_index = index

                elides += [last_index + 1 != len(sequence)]
                matched_words = tuple(matched_words)
                elides = tuple(elides)

                if (matched_words, elides) not in rollups:
                    rollups[(matched_words, elides)] = 0

                rollups[(matched_words, elides)] += 1

        return SequenceRollup([SequenceMatch(key[0], key[1], value) for key, value in sorted(rollups.items(), key=lambda item: item[1], reverse=True)])

    def find_matches(self, tolerance, first_only, predicates):
        check.check_instance(predicates, Predicates)
        # predicates: list of dicts, keyed by rnn part keys to lists of (axis, value) features
        #             [ {(cell, 0): [ (0, 0.5), (22, -0.02), ... ] }, ... ]
        matched_activations = None
        matched_sequences = None

        for level, predicate in predicates.levels():
            matches = None

            for key, features in predicate.items():
                found_sequences = set()
                found_indices = {}
                first_feature = next(iter(features.items()))

                for sequence, index, *point in self._candidates(key, first_feature, tolerance, matched_sequences):
                    point = tuple(point)
                    candidate_point = [point[axis] for axis, target in features.items()]
                    target_point = [target for axis, target in features.items()]
                    within, distance = self._measure(candidate_point, target_point, tolerance)

                    if within:
                        found_sequences.add(sequence)

                        if sequence not in found_indices:
                            found_indices[sequence] = set()

                        found_indices[sequence].add(index)

                if matched_sequences is None:
                    matched_sequences = found_sequences
                    logging.debug("initially matched sequences: %d" % len(matched_sequences))
                else:
                    matched_sequences.intersection_update(found_sequences)
                    logging.debug("subsequently matched sequences: %d" % len(matched_sequences))

                if matches is None:
                    matches = found_indices
                else:
                    next_matches = {}

                    for sequence in matches.keys():
                        if sequence in found_indices:
                            next_matches[sequence] = matches[sequence].intersection(found_indices[sequence])

                    matches = next_matches

            if matched_activations is None:
                matched_activations = {}

                for sequence, indices in matches.items():
                    matched_activations[sequence] = [indices]
            else:
                removes = set()

                for sequence in matched_activations.keys():
                    if sequence in matches:
                        matched_activations[sequence] += [matches[sequence]]
                    else:
                        removes.add(sequence)

                for remove in removes:
                    del matched_activations[remove]

                    if remove in matched_sequences:
                        matched_sequences.remove(remove)

        matches = []

        for sequence, requirements in matched_activations.items():
            #logging.debug("searching for paths through: %s\n  %s" % (" ".join(sequence), requirements))
            paths = monotonic_paths(requirements, len(sequence), first_only)
            #logging.debug("found %d paths" % (len(paths)))

            for path in paths:
                matches += [(sequence, path)]

        return matches

    def _candidates(self, key, axis_target, tolerance, matched_sequences):
        # IPC to ensure sql stays in the thread it was created it.
        request = (key, axis_target, tolerance, matched_sequences)
        request_id = uuid.uuid4()
        self.requests.put((request_id, request))

        while request_id not in self.responses:
            time.sleep(1)

        response = self.responses[request_id]
        del self.responses[request_id]

        if response is None:
            raise RuntimeError("error communicating with sqlite db - try again")
        elif response == "shortcircuit":
            return []

        return response

    def _measure(self, candidate, target, tolerance):
        check.check_lte(check.check_gte(tolerance, 0), 1)
        deltas = geometry.deltas(candidate, target)
        distance = geometry.hypotenuse(deltas)
        return all([part < tolerance for part in deltas]), distance


def monotonic_paths(requirements, length, first_only):
    d = {}

    # initialization
    for i in range(-1, length):
        d[(i, -1)] = set([()])

    # recurrence
    for r in range(len(requirements)):
        found = None

        for i in range(length):
            if found is not None:
                d[(i, r)] = found
            else:
                if i > 0:
                    previous = [m for m in d[(i - 1, r)]]
                else:
                    previous = []

                candidates = [s for s in requirements[r] if s <= i]

                if len(candidates) > 0:
                    q = max(candidates)
                    additions = [m + (q,) for m in d[q - 1, r - 1]]
                else:
                    additions = []

                d[(i, r)] = set(previous + additions)

                if first_only and len(d[(i, r)]) > len(previous):
                    found = d[(i, r)]

    # termination
    return d[(length - 1, len(requirements) - 1)]

