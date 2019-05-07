
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
import pdb
import queue
import random
import resource
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import string
import sys
import threading
import time

from ml import base as mlbase
from ml import model
from ml import nlp
from nnwd import data
from nnwd import geometry
from nnwd import latex
from nnwd.models import Timestep, WeightExplain, WeightDetail, HiddenState, LabelDistribution, SequenceRollup, SequenceMatch, Estimate
from nnwd import parameters
from nnwd import pickler
from nnwd import reduction
from nnwd import rnn
from nnwd import semantic
from nnwd import sequential
from nnwd import view
from pytils.log import setup_logging, user_log
from pytils import adjutant, check


ActivationPoint = collections.namedtuple("ActivationPoint", ["sequence", "expectation", "prediction", "part", "layer", "index", "point"])
MatchPoint = collections.namedtuple("MatchPoint", ["distance", "word", "index", "prediction", "expectation"])


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
    def __init__(self, data_dir, sequential_dir, buckets_dir, encoding_dir):
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
            self.output_mapping = lambda output: pos_mapping[output] if output in pos_mapping else "NN"
            self.colour_mapping = pos_colour_mapping()
            self.sort_key = lambda key_value: -key_value[1]
        else:
            self.outputs = data.get_outputs(self.data_dir)
            self.output_mapping = lambda output: output
            self.colour_mapping = sa_colour_mapping()
            self.sort_key = lambda key_value: sa.sentiment_sort_key(key_value[1])

        self.bucket_mappings = reduction.get_learned_buckets(self.buckets_dir)

        self.lstm = sequential.model_for(self.data_dir)
        sequential.load_model(self.lstm, self.sequential_dir)

        hyper_parameters = model.HyperParameters() \
            .layers(parameters.SEM_LAYERS) \
            .width(parameters.SEM_WIDTH)
        self.predictor = semantic.model_for(self.data_dir, lambda s, i, o: model.Ffnn(s, i, o, hyper_parameters))
        semantic.load_model(self.predictor, self.encoding_dir)

    #@functools.lru_cache()
    def stepwise(self, sequence):
        if len(sequence) == 0:
            return rnn.Stepwise(self.lstm, "root", handle_unknown=True)
        else:
            return self.stepwise(sequence[:-1]).next_stepwise(sequence[-1])

    def query_lstm(self, sequence):
        stepwise_lstm = self.stepwise(tuple(sequence[:-1]))
        word = sequence[-1]
        resolved_word = self.words.decode(self.words.encode(word, True))
        result, instruments = stepwise_lstm.query(word, view.INSTRUMENTS)
        return resolved_word, result, instruments

    def _generate_activation_data(self):
        if self.is_lm():
            # This is a language modelling lm task.
            prediction_fn = lambda y, i: y[i][0]
        else:
            # This is a sentiment analysis sa task.
            prediction_fn = lambda y, i: y

        activation_data_path = os.path.join(self.save_dir, "activation-data")

        if os.path.exists(activation_data_path):
            logging.debug("Activation data already generated.")
        else:
            logging.debug("Generating activation data.")
            activation_data = []
            data_quarter = max(1, int(len(self.train_xys) / 4.0))

            for j, xy in enumerate(self.train_xys):
                if j % data_quarter == 0 or j + 1 == len(self.train_xys):
                    logging.debug("%d%% through.." % int((j + 1) * 100 / len(self.train_xys)))

                stepwise_lstm = self.lstm.stepwise(handle_unknown=True)
                sequence = tuple(xy.x)

                for i, word_pos in enumerate(xy.x):
                    result, instruments = stepwise_lstm.step(word_pos[0], view.INSTRUMENTS)
                    #part = "embedding"
                    #layer = 0
                    #point = tuple(instruments[part])
                    #activation_data += [ActivationPoint(sequence=sequence, expectation=prediction_fn(xy.y, i), prediction=result.prediction, part=part, layer=layer, index=i, point=point)]

                    for part, layer in view.part_layers():
                        point = tuple(instruments[part][layer])
                        activation_data += [ActivationPoint(sequence=sequence, expectation=prediction_fn(xy.y, i), prediction=result.prediction, part=part, layer=layer, index=i, point=point)]

            pickler.dump(activation_data, activation_data_path)
            logging.debug("Generated activation data: %d." % len(activation_data))
            del activation_data

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

            #print(adjutant.dict_as_str(mapping))

        #choice = input("choice (05, 10, 20, 30, 50): ")
        return colour_embeddings[50]

    def mapped_output(self, output):
        resolved = self.outputs.decode(self.outputs.encode(output, True))
        return self.output_mapping(resolved)

    def output_colour(self, output):
        return self.colour_mapping[self.mapped_output(output)]

    def rgb(self, colour):
        return "rgb(%d, %d, %d)" % colour

    def compute_point_abstractions(self, points):
        reductions = self.dimensionality_reduce(points)
        predictions = self.predict_distributions(points)
        colours = self.fit_colours(points, predictions)
        return reductions, colours, predictions

    def dimensionality_reduce(self, points):
        return {key: reduction.reduce(self.bucket_mappings[key], point) for key, point in points.items()}

    def predict_distributions(self, points):
        xs = [semantic.as_input(key, point) for key, point in points.items()]
        results = self.predictor.evaluate(xs)
        distribution_predictions = {}

        for i, x in enumerate(xs):
            part, layer, _ = x
            ordered_predictions = [item[0] for item in sorted(results[i].distribution().items(), key=lambda item: item[1], reverse=True)]
            distribution_predictions[self.encode_key(part, layer)] = {prediction: results[i].distribution()[prediction] for prediction in ordered_predictions[:parameters.TOP_PREDICTIONS]}

        return distribution_predictions

    def fit_colours(self, points, predictions):
        colours = {}

        for key, point in points.items():
            output_colourings = {}

            for output, probability in predictions[key].items():
                self.mapped_output(output)
                colour = self.output_colour(output)
                output_colourings[output] = (colour, probability)

            interpolation_points = {}

            for output, colour_probability in sorted(output_colourings.items(), key=lambda item: item[1][1], reverse=True)[:2]:
                colour, probability = colour_probability
                interpolation_points[colour] = (output, probability)

            if len(interpolation_points) == 1:
                colours[key] = "rgb(%d, %d, %d)" % next(iter(interpolation_points.keys()))
            else:
                point_a, point_b = [item for item in interpolation_points.items()]
                distance = geometry.distance(point_a[0], point_b[0])
                # Not a typo: we want to invert their probabilities so that the most likely prediction gets the smallest distance, and visa versa.
                #                            v              v              v              v
                pdist = mlbase.regmax({point_a[1][0]: point_b[1][1], point_b[1][0]: point_a[1][1]})
                fit = geometry.fit_proportion((point_a[0], point_b[0]), (pdist[point_a[1][0]], pdist[point_b[1][0]]))
                colours[key] = "rgb(%d, %d, %d)" % tuple([round(i) for i in fit])

        return colours

    def weights(self, sequence):
        last_word, result, instruments = self.query_lstm(sequence)
        points = {}

        for part, layer in view.part_layers():
            # TODO
            if part == "embedding":
                embedding_key = view.encode_key(part, layer)
                points[self.encode_key(part, layer)] = instruments[part]
            else:
                points[self.encode_key(part, layer)] = instruments[part][layer]

        point_reductions, point_colours, point_predictions = self.compute_point_abstractions(points)
        embedding_name = self.latex_name(len(sequence) - 1, "embedding")
        embedding_name_no_t = self.latex_name_no_t("embedding")
        embedding = HiddenState(embedding_name, embedding_name_no_t, point_reductions[embedding_key], colour=point_colours[embedding_key], predictions=self.prediction_distribution(point_predictions[embedding_key]))
        units = self.make_lstm_units(len(sequence) - 1, point_reductions, point_colours, point_predictions)
        softmax_name = self.latex_name(len(sequence) - 1, "softmax")
        softmax = LabelDistribution(softmax_name, result.distribution, self.sort_key, parameters.OUTPUT_WIDTH, lambda output: self.rgb(self.output_colour(output)))
        return Timestep(embedding, units, softmax, len(sequence) - 1, last_word, result.prediction)

    def weight_detail(self, sequence, part, layer):
        last_word, result, instruments = self.query_lstm(sequence)
        point = instruments[part]

        if layer is not None:
            point = point[layer]

        if part.endswith("_gates"):
            min_max = (0, 1)
        else:
            min_max = (None, None)

        # This is the regular process by which a hidden_state is served out with its various reductions/abstractions.
        key = self.encode_key(part, layer)
        keyed_point = {key: point}
        reduction, colour, prediction = self.compute_point_abstractions(keyed_point)
        name = self.latex_name(len(sequence) - 1, part, layer)
        name_no_t = self.latex_name_no_t(part, layer)
        hidden_state = HiddenState(name, name_no_t, reduction[key], min_max, colour[key], self.prediction_distribution(prediction[key]))
        back_links = {}
        reorganized_point = [[] for i in range(len(point))]
        i = 0

        for bucket, dimensions in self.bucket_mappings[key].items():
            for dimension in dimensions:
                assert i not in back_links, "%d already in %s" % (i, back_links)
                back_links[i] = bucket
                reorganized_point[i] = point[dimension]
                i += 1

        # This is the full point.
        full_hidden_state = HiddenState(name, name_no_t, reorganized_point, min_max, None, None)

        return WeightDetail(hidden_state, full_hidden_state, back_links)

    def encode_key(self, part, layer=None):
        return "%s-%d" % (part, 0 if layer is None else layer)

    def decode_key(self, key):
        result = key.split("-")

        if len(result) == 2:
            return (result[0], int(result[1]))
        else:
            raise ValueError("invalid decode of '%s': %s" % (key, result))

    def prediction_distribution(self, prediction):
        if prediction is None:
            return None

        return LabelDistribution(None, prediction, self.sort_key, colour_fn=lambda output: self.rgb(self.output_colour(output)))

    def make_lstm_units(self, timestep, point_reductions, point_colours, point_predictions):
        units = {}

        for part in view.LSTM_PARTS:
            units[part] = {}

            for layer in range(parameters.LAYERS):
                if part.endswith("_gates"):
                    min_max = (0, 1)
                else:
                    min_max = (None, None)

                key = self.encode_key(part, layer)
                name = self.latex_name(timestep, part, layer)
                name_no_t = self.latex_name_no_t(part, layer)
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


class QueryEngine:
    TOP = 25

    def __init__(self):
        self.thresholds = {}
        self.save_dir = "moot"
        #self._background_setup = threading.Thread(target=self._setup)
        #self._background_setup.daemon = True
        #self._background_setup.start()

    def _setup(self):
        activation_data_path = os.path.join(self.save_dir, "activation-data")
        logging.debug("Waiting on activation data.")

        while not os.path.exists(activation_data_path):
            time.sleep(1)

        logging.debug("Processing activation data for query engine.")
        self.units = {}
        self.sequence_units = {}

        for i, activation_point in enumerate(pickler.load(activation_data_path)):
            if i % 1000000 == 0:
                logging.debug("At the %dMth instance." % int(i / 1000000))

            sequence = tuple(activation_point.sequence)
            unit = (activation_point.part, activation_point.layer)
            sequence_unit = (sequence,) + unit

            if sequence_unit not in self.sequence_units:
                self.sequence_units[sequence_unit] = []

            if unit not in self.units:
                self.units[unit] = {}

            self.sequence_units[sequence_unit] += [activation_point]

            for axis, value in enumerate(activation_point.point):
                if axis not in self.units[unit]:
                    self.units[unit][axis] = []

                self.units[unit][axis] += [activation_point]

        for unit, subd in self.units.items():
            for axis, points in subd.items():
                self.units[unit][axis] = sorted(points, key=lambda ap: ap.point[axis])

        user_log.info("Setup QueryEngine.")

    def find_estimate(self, tolerance, predicates):
        matches = self.find_matches(tolerance, True, predicates)
        lower = len(matches)
        return Estimate(lower=lower, upper=None if lower > 10 else lower * 10)

    def find(self, tolerance, predicates):
        matches = self.find_matches(tolerance, False, predicates)
        rollups = {}

        for sequence, result in matches:
            matched_words = []
            elides = []
            last_index = None

            for match_point in result:
                matched_words += [sequence[match_point.index]]

                if last_index is None:
                    elides += [match_point.index != 0]
                elif last_index + 1 == match_point.index:
                    elides += [False]
                else:
                    elides += [True]

                last_index = match_point.index

            elides += [last_index + 1 != len(sequence)]
            matched_words = tuple(matched_words)
            elides = tuple(elides)

            if (matched_words, elides) not in rollups:
                rollups[(matched_words, elides)] = 0

            rollups[(matched_words, elides)] += 1

        return SequenceRollup([SequenceMatch(key[0], key[1], value) for key, value in sorted(rollups.items(), key=lambda item: item[1], reverse=True)[:QueryEngine.TOP]])

    def find_matches(self, tolerance, first_only, predicates):
        required_level_units = []
        matched_activations = {}
        matched_sequences = None

        for level, predicate in enumerate(predicates):
            for unit, features in predicate.items():
                level_unit = (level,) + unit
                required_level_units += [level_unit]
                found = set()

                for activation_point in self._candidates(unit, next(iter(features)), tolerance, matched_sequences):
                    candidate_point = [activation_point.point[axis] for axis, target in features]
                    target_point = [target for axis, target in features]
                    within, distance = self._measure(candidate_point, target_point, tolerance)

                    if within:
                        sequence = tuple(activation_point.sequence)
                        found.add(sequence)

                        if sequence not in matched_activations:
                            matched_activations[sequence] = {}

                        if level_unit not in matched_activations[sequence]:
                            matched_activations[sequence][level_unit] = []

                        matched_activations[sequence][level_unit] += [(distance, activation_point)]

                if matched_sequences is None:
                    matched_sequences = found
                    logging.debug("matched_sequences: %d" % len(matched_sequences))
                else:
                    matched_sequences.intersection_update(found)
                    logging.debug("matched_sequences: %d" % len(matched_sequences))

        logging.debug("matched_activation keys: %s" % str([key for key in matched_activations.keys()]))
        logging.debug("matched_activation units: %s" % str([subd.keys() for subd in matched_activations.values()]))
        matches = []

        for sequence, level_unit_instances in matched_activations.items():
            results = self.find_match_points(required_level_units, level_unit_instances, 0, None, first_only)

            if len(results) > 0:
                logging.debug("matched sequence %d times: %s" % (len(results), " ".join(sequence)))

            for result in results:
                matches += [(sequence, result)]

        return matches

    def find_match_points(self, required_level_units, level_unit_instances, requirement_index, match_index, first_only):
        current_level_unit = required_level_units[requirement_index]

        # This case doesn't even have instances across all the constraining level_units (level, part, layer) - definitely can't be satisified.
        if len(level_unit_instances) < len(required_level_units):
            return []

        if requirement_index > 0:
            previous_level = required_level_units[requirement_index - 1][0]
            assert previous_level <= current_level_unit[0]

            if previous_level == current_level_unit[0]:
                monotonically_increasing = lambda ap: ap.index == match_index
            else:
                monotonically_increasing = lambda ap: ap.index > match_index
        else:
            monotonically_increasing = lambda ap: True

        results = []

        for instance in level_unit_instances[current_level_unit]:
            distance, activation_point = instance

            if match_index is None or monotonically_increasing(activation_point):
                word = activation_point.sequence[activation_point.index]

                # If we're at the final constraint.
                if requirement_index + 1 == len(required_level_units):
                    # Found
                    results += [[MatchPoint(distance=distance, word=word, index=activation_point.index, prediction=activation_point.prediction, expectation=activation_point.expectation)]]

                    if first_only:
                        break
                else:
                    sub_results = self.find_match_points(required_level_units, level_unit_instances, requirement_index + 1, activation_point.index, first_only)

                    for r in sub_results:
                        # Note, we don't need to worry about only adding to results once if first_only, because by definition len(sub_results) must only be 0 or 1.
                        results += [[MatchPoint(distance=distance, word=word, index=activation_point.index, prediction=None, expectation=None)] + r]

                    if first_only and len(sub_results) > 0:
                        break

        return results

    def _candidates(self, unit, axis_target, tolerance, matched_sequences):
        if matched_sequences is None:
            axis, target = axis_target
            sorted_activation_points = self.units[unit][axis]
            bottom_index = self._binary_search(axis, target - tolerance, sorted_activation_points, True)
            top_index = self._binary_search(axis, target + tolerance, sorted_activation_points, False)
            return self.units[unit][axis][bottom_index:top_index + 1]
        else:
            return adjutant.flat_map([self.sequence_units[(sequence,) + unit] for sequence in matched_sequences])

    def _binary_search(self, axis, target, activation_points, find_lower):
        lower = 0
        upper = len(activation_points) - 1
        found = upper if activation_points[upper].point[axis] == target else None

        while found is None:
            if activation_points[lower].point[axis] > target:
                found = lower
            elif activation_points[upper].point[axis] < target:
                found = upper
            else:
                current = int((upper + lower) / 2.0)
                observation = activation_points[current].point[axis]

                if observation == target:
                    found = current
                elif observation < target:
                    if lower == current:
                        if activation_points[upper].point[axis] <= target:
                            found = upper
                        else:
                            found = lower if find_lower else upper
                    else:
                        lower = current
                else:
                    upper = current

        direction = -1 if find_lower else 1

        while found + direction < len(activation_points) and activation_points[found + direction].point[axis] == target:
            found += direction

        return found

    def _measure(self, candidate, target, tolerance):
        check.check_lte(check.check_gte(tolerance, 0), 1)
        deltas = geometry.deltas(candidate, target)
        distance = geometry.hypotenuse(deltas)
        return all([part < tolerance for part in deltas]), distance

