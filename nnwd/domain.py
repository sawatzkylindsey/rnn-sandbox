
import collections
import functools
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
import string
import threading
import time

from ml import base as mlbase
from ml import nlp
from ml import nn as ffnn
from nnwd import geometry
from nnwd import latex
from nnwd.models import Timestep, WeightExplain, WeightDetail, HiddenState, LabelDistribution, SequenceRollup, SequenceMatch, Estimate
from nnwd import pickler
from nnwd import rnn
from pytils.log import setup_logging, user_log
from pytils import adjutant, check


RESUME_DIR = ".resume"
MARKER = "-marker"
ActivationPoint = collections.namedtuple("ActivationPoint", ["sequence", "expectation", "prediction", "part", "layer", "index", "point"])
MatchPoint = collections.namedtuple("MatchPoint", ["distance", "word", "index", "prediction", "expectation"])


def create_sa(task_form, corpus_stream_fn, epochs, verbose):
    train_xys_file = os.path.join(RESUME_DIR, "xys.train.pickle")
    validation_xys_file = os.path.join(RESUME_DIR, "xys.validation.pickle")
    test_xys_file = os.path.join(RESUME_DIR, "xys.test.pickle")
    sentiments_file = os.path.join(RESUME_DIR, "sentiments.pickle")
    words_file = os.path.join(RESUME_DIR, "words.pickle")

    if os.path.exists(words_file + MARKER):
        train_xys = [xy for xy in pickler.load(train_xys_file)]
        validation_xys = [xy for xy in pickler.load(validation_xys_file)]
        test_xys = [xy for xy in pickler.load(test_xys_file)]
        sentiments = set([sentiment for sentiment in pickler.load(sentiments_file)])
        words = set([word for word in pickler.load(words_file)])
    else:
        train_xys = []
        validation_xys = []
        test_xys = []
        sentiments = set()
        words = set()

        for triple in corpus_stream_fn():
            sentiment = get_sentiment(triple[2])
            sentiments.add(sentiment)
            xy = ([(word, None) for word in triple[1]], sentiment)

            if triple[0] == "train":
                for word in triple[1]:
                    words.add(word)

                train_xys += [xy]
            elif triple[0] == "dev":
                validation_xys += [xy]
            else:
                test_xys += [xy]

        pickler.dump(train_xys, train_xys_file)
        pickler.dump(validation_xys, validation_xys_file)
        pickler.dump(test_xys, test_xys_file)
        pickler.dump(sorted(sentiments, key=sentiment_sort_key), sentiments_file)
        pickler.dump([word for word in words], words_file)

        with open(words_file + MARKER, "w") as fh:
            fh.write("noop")

    #printer = lambda sequences: "\n".join([" ".join([str(word_pos) for word_pos in i]) for i in sequences])
    #print(printer(train_xys))
    #print(printer(validation_xys))
    #print(printer(test_xys))
    logging.debug("data sets (train, validation, test): %d, %d, %d" % (len(train_xys), len(validation_xys), len(test_xys)))
    words = mlbase.Labels(words.union(set([mlbase.BLANK])), unknown=nlp.UNKNOWN)
    #print(words)
    sentiments = mlbase.Labels(sentiments)
    return words, NeuralNetwork(task_form, words, sentiments, None, None, train_xys, epochs, validation_xys, test_xys, lambda item: sentiment_sort_key(item[0]))


def get_sentiment(value):
    # [0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]
    # for very negative, negative, neutral, positive, very positive, respectively.
    if value <= 0.2:
        return "very negative"
    elif value <= 0.4:
        return "negative"
    elif value <= 0.6:
        return "neutral"
    elif value <= 0.8:
        return "positive"
    else:
        return "very positive"


def sentiment_sort_key(sentiment):
    if sentiment == "very negative":
        return 4
    elif sentiment == "negative":
        return 3
    elif sentiment == "neutral":
        return 2
    elif sentiment == "positive":
        return 1
    else:
        return 0


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


def create_lm(task_form, corpus_stream_fn, epochs, verbose):
    train_xys_file = os.path.join(RESUME_DIR, "xys.train.pickle")
    validation_xys_file = os.path.join(RESUME_DIR, "xys.validation.pickle")
    test_xys_file = os.path.join(RESUME_DIR, "xys.test.pickle")
    pos_tags_file = os.path.join(RESUME_DIR, "pos_tags.pickle")
    pos_mapping_file = os.path.join(RESUME_DIR, "pos_mapping.pickle")
    words_file = os.path.join(RESUME_DIR, "words.pickle")

    if os.path.exists(words_file + MARKER):
        train_xys = [xy for xy in pickler.load(train_xys_file)]
        validation_xys = [xy for xy in pickler.load(validation_xys_file)]
        test_xys = [xy for xy in pickler.load(test_xys_file)]
        pos_tags = set([word for word in pickler.load(pos_tags_file)])
        pos_mapping = {item[0]: item[1] for item in pickler.load(pos_mapping_file)}
        words = set([word for word in pickler.load(words_file)])
    else:
        xys = [sentence for sentence in corpus_stream_fn()]
        random.shuffle(xys)
        split_1 = int(len(xys) * 0.8)
        split_2 = split_1 + int(len(xys) * 0.1)
        train_xys = xys[:split_1]
        validation_xys = xys[split_1:split_2]
        test_xys = xys[split_2:]
        # Words are actually only the subset from the training data.
        words = set(adjutant.flat_map([[word_pos[0] for word_pos in sentence] for sentence in train_xys]))
        pos_tags = set(adjutant.flat_map([[word_pos[1] for word_pos in sentence] for sentence in train_xys]))
        pickler.dump(train_xys, train_xys_file)
        pickler.dump(validation_xys, validation_xys_file)
        pickler.dump(test_xys, test_xys_file)
        pickler.dump([pos for pos in pos_tags], pos_tags_file)
        pickler.dump([word for word in words], words_file)
        pos_counts = {}

        for sentence in train_xys:
            for word_pos in sentence:
                if word_pos[0] not in pos_counts:
                    pos_counts[word_pos[0]] = {}

                if word_pos[1] not in pos_counts[word_pos[0]]:
                    pos_counts[word_pos[0]][word_pos[1]] = 0

                pos_counts[word_pos[0]][word_pos[1]] += 1

        pos_mapping = {}

        for word, counts in pos_counts.items():
            pos_count = sorted(counts.items(), key=lambda item: item[1], reverse=True)[0]
            pos_mapping[word] = pos_count[0]

        pickler.dump([item for item in pos_mapping.items()], pos_mapping_file)

        with open(words_file + MARKER, "w") as fh:
            fh.write("noop")

    #printer = lambda sequences: "\n".join([" ".join([str(word_pos) for word_pos in i]) for i in sequences])
    #print(printer(train_xys))
    #print(printer(validation_xys))
    #print(printer(test_xys))
    logging.debug("data sets (train, validation, test): %d, %d, %d" % (len(train_xys), len(validation_xys), len(test_xys)))
    words = mlbase.Labels(words.union(set([mlbase.BLANK])), unknown=nlp.UNKNOWN)
    #print(words)
    return words, NeuralNetwork(task_form, words, None, pos_tags, pos_mapping, xy_sequence(train_xys), epochs, xy_sequence(validation_xys), xy_sequence(test_xys), lambda item: -item[1])


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


def xy_sequence(xys):
    return [(sequence[:-1], sequence[1:]) for sequence in xys if len(sequence) > 1]


class NeuralNetwork:
    LAYERS = 2
    HIDDEN_WIDTH = 50
    EMBEDDING_WIDTH = 50
    OUTPUT_WIDTH = 5
    HIDDEN_REDUCTION = 10
    EMBEDDING_REDUCTION = 10
    TOP_PREDICTIONS = 3
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

    def __init__(self, task_form, words, outputs, pos_tags, pos_mapping, train_xys, epoch_threshold, validation_xys, test_xys, sort_key):
        self.task_form = task_form
        self.words = check.check_instance(words, mlbase.Labels)
        self.outputs = outputs
        self.pos_tags = pos_tags
        self.pos_mapping = pos_mapping
        self.train_xys = [mlbase.Xy(*pair) for pair in train_xys]
        self.epoch_threshold = epoch_threshold
        self.validation_xys = [mlbase.Xy(*pair) for pair in validation_xys]
        self.test_xys = [mlbase.Xy(*pair) for pair in test_xys]

        if self.is_lm():
            if self.has_pos():
                self.colour_mapping = pos_colour_mapping()
            else:
                self.colour_mapping = parens_colour_mapping()
        else:
            self.colour_mapping = sa_colour_mapping()

        self.sort_key = sort_key
        self.setup_complete = False
        self.embedding_padding = tuple([0] * max(0, NeuralNetwork.HIDDEN_WIDTH - NeuralNetwork.EMBEDDING_WIDTH))
        self.hidden_padding = tuple([0] * max(0, NeuralNetwork.EMBEDDING_WIDTH - NeuralNetwork.HIDDEN_WIDTH))
        self._background_setup = threading.Thread(target=self._setup)
        self._background_setup.daemon = True
        self._background_setup.start()

    def is_lm(self):
        return self.task_form[0] == "lm"

    def has_pos(self):
        return self.task_form[0] == "lm" and self.task_form[1] == "pos"

    def _setup(self):
        self._train_rnn()
        #self._generate_activation_data()
        # Sets setup_complete
        self._train_predictor()
        user_log.info("Setup NeuralNetwork")

    def is_setup(self):
        return self.setup_complete
        #return not self._background_setup.is_alive()

    def _train_rnn(self):
        if self.is_lm():
            self.lstm = rnn.RnnLm(NeuralNetwork.LAYERS, NeuralNetwork.HIDDEN_WIDTH, NeuralNetwork.EMBEDDING_WIDTH, self.words)
        else:
            # This is a sentiment analysis sa task.
            self.lstm = rnn.RnnSa(NeuralNetwork.LAYERS, NeuralNetwork.HIDDEN_WIDTH, NeuralNetwork.EMBEDDING_WIDTH, self.words, self.outputs)

        lstm_dir = os.path.join(RESUME_DIR, "lstm")

        if os.path.exists(lstm_dir):
            logging.debug("Loading existing lstm parameters.")
            self.lstm.load(lstm_dir)
        else:
            logging.debug("Training lstm parameters.")
            score = self.lstm.test(self.validation_xys)
            logging.debug("Baseline score (random initialized weights): %s" % score)
            score_validation = None
            best_score = None
            best_loss = None
            batch = 32
            arc = -1
            version = -1
            max_arc = 4

            while arc < max_arc:
                arc += 1
                logging.debug("train lstm arc %d (batch %d)" % (arc, batch))
                loss, score = self.lstm_train_loop(batch, self.epoch_threshold, True)
                logging.debug("train lstm arc %d (batch %d): (loss, score) (%s, %s)" % (arc, batch, loss, score))

                if best_score is None or score > best_score:
                    best_score = score
                    best_loss = loss
                    score_validation = score
                    version += 1
                    self.lstm.save(lstm_dir, version, True)
                elif arc == max_arc:
                    self.lstm.load(lstm_dir)
                else:
                    best_loss = None
                    self.lstm.load(lstm_dir, version)
                    mini_epochs = max(1, int(self.epoch_threshold * 0.2))
                    smaller_batch = max(8, int(batch * 0.8))
                    smaller_loss, smaller_score = self.lstm_train_loop(smaller_batch, mini_epochs, False)
                    logging.debug("mini train lstm smaller (batch %d): (loss, score) (%s, %s)" % (smaller_batch, smaller_loss, smaller_score))
                    self.lstm.save(lstm_dir, "smaller-%d" % version)
                    self.lstm.load(lstm_dir, version)
                    bigger_batch = int(batch * 1.2)
                    bigger_loss, bigger_score = self.lstm_train_loop(bigger_batch, mini_epochs, False)
                    logging.debug("mini train lstm bigger (batch %d): (loss, score) (%s, %s)" % (bigger_batch, bigger_loss, bigger_score))
                    self.lstm.save(lstm_dir, "bigger-%d" % version)
                    self.lstm.load(lstm_dir, version)

                    if bigger_score > score_validation:
                        # If the bigger batches are getting a better score, choose it (maybe the smaller is as well, but bigger usually means more general learning).
                        batch = bigger_batch
                        self.lstm.mark_latest(lstm_dir, "bigger-%d" % version)
                        self.lstm.load(lstm_dir)
                    elif smaller_score > score_validation:
                        # Certainly choose it
                        batch = smaller_batch
                        self.lstm.mark_latest(lstm_dir, "smaller-%d" % version)
                        self.lstm.load(lstm_dir)
                    else:
                        # Neither are getting better results.  Which is best among the choices?
                        if bigger_score > smaller_score:
                            batch = bigger_batch
                        else:
                            batch = smaller_batch

        logging.debug("Calculating final validation score.")
        score_validation = self.lstm.test(self.validation_xys, False)
        logging.debug("Calculating final test score.")
        score_test = self.lstm.test(self.test_xys, True)
        logging.debug("(v, t): (%s, %s)" % (score_validation, score_test))

    def lstm_train_loop(self, batch, epoch_threshold, debug):
        training_parameters = mlbase.TrainingParameters() \
            .batch(batch) \
            .epochs(epoch_threshold) \
            .convergence(False) \
            .debug(debug)
        loss = self.lstm.train(self.train_xys, training_parameters)
        score = self.lstm.test(self.validation_xys)
        return loss, score

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
        result, instruments = stepwise_lstm.query(word, NeuralNetwork.INSTRUMENTS)
        return resolved_word, result, instruments

    def _train_predictor(self):
        part_labels = mlbase.Labels(set(NeuralNetwork.INSTRUMENTS))
        layer_labels = mlbase.Labels(set(range(NeuralNetwork.LAYERS)))
        hidden_vector = mlbase.VectorField(max(NeuralNetwork.HIDDEN_WIDTH, NeuralNetwork.EMBEDDING_WIDTH))
        predictor_input = mlbase.ConcatField([part_labels, layer_labels, hidden_vector])

        if self.is_lm():
            if self.has_pos():
                predictor_output = mlbase.Labels(self.pos_tags)
            else:
                predictor_output = mlbase.Labels(set(self.words.labels()))
        else:
            predictor_output = mlbase.Labels(set(self.outputs.labels()))

        hyper_parameters = ffnn.HyperParameters() \
            .layers(2) \
            .width(int((len(predictor_input) + len(predictor_output)) / 2.0))
        self.predictor = ffnn.Model("predictor", hyper_parameters, predictor_input, predictor_output)
        predictor_dir = os.path.join(RESUME_DIR, "predictor")

        if os.path.exists(predictor_dir):
            logging.debug("Loading existing predictor parameters.")
            self.predictor.load(predictor_dir)
            self.setup_complete = True
        else:
            predictor_xys = self._get_predictor_data()
            logging.debug("Training predictor parameters.")
            training_parameters = mlbase.TrainingParameters() \
                .epochs(100) \
                .batch(32)
            # Technically not complete yet, but with the predictor setup it can start answering queries.
            self.setup_complete = True
            loss = self.predictor.train(predictor_xys, training_parameters)
            logging.debug("train predictor %s" % loss)
            self.predictor.save(predictor_dir)
            del predictor_xys

        self._test_predictor()

    def _test_predictor(self):
        distributions = []

        for j, xy in enumerate(self.test_xys):
            stepwise_lstm = self.lstm.stepwise(False)

            for i, word_pos in enumerate(xy.x):
                xs = []
                result, instruments = stepwise_lstm.step(word_pos[0], NeuralNetwork.INSTRUMENTS)
                x = ("embedding", 0, tuple(instruments["embedding"]) + self.embedding_padding)
                xs += [x]

                for part in NeuralNetwork.LSTM_PARTS:
                    for layer in range(NeuralNetwork.LAYERS):
                        point = tuple(instruments[part][layer]) + self.hidden_padding
                        x = (part, layer, point)
                        xs += [x]

                for result in self.predictor.evaluate(xs):
                    distributions += [result.distribution]

        logging.debug("Predictor test distributions: %d." % len(distributions))
        pickler.dump(distributions, os.path.join(RESUME_DIR, "sem-distributions.pickle"))

    def _get_predictor_data(self):
        if self.is_lm():
            if self.has_pos():
                prediction_fn = lambda y, i: y[i][1]
            else:
                prediction_fn = lambda y, i: y[i][0]
        else:
            # This is a sentiment analysis sa task.
            prediction_fn = lambda y, i: y

        predictor_xys = []
        predictor_xys_file = os.path.join(RESUME_DIR, "predictor_xys.pickle")

        if os.path.exists(predictor_xys_file + MARKER):
            logging.debug("Loading existing predictor data.")
            predictor_xys = [xy for xy in pickler.load(predictor_xys_file)]
        else:
            logging.debug("Producing predictor data.")
            data_quarter = max(1, int(len(self.train_xys) / 4.0))

            for j, xy in enumerate(self.train_xys):
                if j % data_quarter == 0 or j + 1 == len(self.train_xys):
                    logging.debug("%d%% through.." % int((j + 1) * 100 / len(self.train_xys)))

                stepwise_lstm = self.lstm.stepwise(False)

                for i, word_pos in enumerate(xy.x):
                    # Set the prediction to that which the lstm has been trained against, not the actual learned prediction (which will be fixed).
                    # For example, consider the two training examples: "the little prince" -> "was" and "the little prince" -> "is".
                    # We need predictor samples for both "was" and "is", but if we use the actual lstm prediction this will fixate on just one of these.
                    prediction = prediction_fn(xy.y, i)
                    result, instruments = stepwise_lstm.step(word_pos[0], NeuralNetwork.INSTRUMENTS)
                    x = ("embedding", 0, tuple(instruments["embedding"]) + self.embedding_padding)
                    train_xy = mlbase.Xy(x, prediction)
                    predictor_xys += [train_xy]

                    for part in NeuralNetwork.LSTM_PARTS:
                        for layer in range(NeuralNetwork.LAYERS):
                            point = tuple(instruments[part][layer]) + self.hidden_padding
                            x = (part, layer, point)
                            train_xy = mlbase.Xy(x, prediction)
                            predictor_xys += [train_xy]

            pickler.dump(predictor_xys, predictor_xys_file)

            with open(predictor_xys_file + MARKER, "w") as fh:
                fh.write("noop")

        logging.debug("Predictor data: %d." % len(predictor_xys))
        return predictor_xys

    def _generate_activation_data(self):
        if self.is_lm():
            # This is a language modelling lm task.
            prediction_fn = lambda y, i: y[i][0]
        else:
            # This is a sentiment analysis sa task.
            prediction_fn = lambda y, i: y

        activation_data_file = os.path.join(RESUME_DIR, "activation_data.pickle")

        if os.path.exists(activation_data_file + MARKER):
            logging.debug("Activation data already generated.")
        else:
            logging.debug("Generating activation data.")
            activation_data = []
            data_quarter = max(1, int(len(self.train_xys) / 4.0))

            for j, xy in enumerate(self.train_xys):
                if j % data_quarter == 0 or j + 1 == len(self.train_xys):
                    logging.debug("%d%% through.." % int((j + 1) * 100 / len(self.train_xys)))

                stepwise_lstm = self.lstm.stepwise(False)
                sequence = tuple(xy.x)

                for i, word_pos in enumerate(xy.x):
                    result, instruments = stepwise_lstm.step(word_pos[0], NeuralNetwork.INSTRUMENTS)
                    part = "embedding"
                    layer = 0
                    point = tuple(instruments[part])
                    activation_data += [ActivationPoint(sequence=sequence, expectation=prediction_fn(xy.y, i), prediction=result.prediction, part=part, layer=layer, index=i, point=point)]

                    for part in NeuralNetwork.LSTM_PARTS:
                        for layer in range(NeuralNetwork.LAYERS):
                            point = tuple(instruments[part][layer])
                            activation_data += [ActivationPoint(sequence=sequence, expectation=prediction_fn(xy.y, i), prediction=result.prediction, part=part, layer=layer, index=i, point=point)]

            pickler.dump(activation_data, activation_data_file)

            with open(activation_data_file + MARKER, "w") as fh:
                fh.write("noop")

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


    def output_colour(self, output):
        if not self.is_setup():
            return None

        colour = self.colour_mapping[output]
        return "rgb(%d, %d, %d)" % colour

    def compute_point_abstractions(self, points):
        reductions = self.dimensionality_reduce(points, NeuralNetwork.HIDDEN_REDUCTION)

        if not self.is_setup():
            colours = {key: None for key, point in points.items()}
            predictions = {key: None for key, point in points.items()}
            return reductions, colours, predictions

        predictions = self.predict_distributions(points)
        colours = self.fit_colours(points, predictions)
        return reductions, colours, predictions

    def dimensionality_reduce(self, points, reduction_size):
        reductions = {}

        for key, endpoints in self.dimensionality_reduce_mapping(points, reduction_size).items():
            reduction = []

            for i, endpoint in sorted(endpoints.items()):
                bucket = points[key][endpoint[0]:endpoint[1]]
                reduction += [sum(bucket) / float(len(bucket))]

            reductions[key] = reduction

        return reductions

    def dimensionality_reduce_mapping(self, points, reduction_size):
        mappings = {}

        for key, point in points.items():
            bucket_size = math.ceil(len(point) / reduction_size)
            endpoints = {}
            offset = 0
            i = 0

            while offset < len(point):
                endpoints[i] = (offset, min(offset + bucket_size, len(point)))
                offset += bucket_size
                i += 1

            mappings[key] = endpoints

        return mappings

    def predict_distributions(self, points):
        xs = [self.decode_key(key) + (tuple(point) + (self.embedding_padding if self.decode_key(key)[0] == "embedding" else self.hidden_padding),) for key, point in points.items()]
        results = self.predictor.evaluate(xs)
        distribution_predictions = {}

        for i, x in enumerate(xs):
            part, layer, _ = x
            ordered_predictions = [item[0] for item in sorted(results[i].distribution.items(), key=lambda item: item[1], reverse=True)]
            distribution_predictions[self.encode_key(part, layer)] = {prediction: results[i].distribution[prediction] for prediction in ordered_predictions[:self.TOP_PREDICTIONS]}

        return distribution_predictions

    def fit_colours(self, points, predictions):
        colours = {}

        for key, point in points.items():
            #most_likely_prediction = sorted(predictions[key].items(), key=lambda item: item[1], reverse=True)[0][0]
            #colours[key] = "rgb(%d, %d, %d)" % self.colour_mapping[most_likely_prediction]
            interpolation_points = {}

            for word, probability in predictions[key].items():
                colour = self.colour_mapping[word]

                if colour not in interpolation_points:
                    interpolation_points[colour] = (word, probability)
                else:
                    if interpolation_points[colour][1] < probability:
                        interpolation_points[colour] = (word, probability)

            if len(interpolation_points) == 1:
                colours[key] = "rgb(%d, %d, %d)" % next(iter(interpolation_points.keys()))
            else:
                maximum_distance = None

                for pair in itertools.combinations([colour for colour in interpolation_points.keys()], 2):
                    distance = geometry.distance(pair[0], pair[1])

                    if maximum_distance is None or distance > maximum_distance:
                        maximum_distance = distance

                lowest_probability = min([p for w, p in interpolation_points.values()])
                highest_probability = max([p for w, p in interpolation_points.values()])
                ceiling = highest_probability * 1.1
                maximum_domain = highest_probability + lowest_probability
                pdist = mlbase.regmax({w: ceiling - p for w, p in interpolation_points.values()})
                prediction_distances = [(w, maximum_distance + (p * maximum_distance / maximum_domain)) for w, p in pdist.items()]
                fit, _ = geometry.fit_point([self.colour_mapping[item[0]] for item in prediction_distances], [item[1] for item in prediction_distances], epsilon=0.1, visualize=False)
                colours[key] = "rgb(%d, %d, %d)" % tuple([round(i) for i in fit])

        return colours

    def weights(self, sequence):
        last_word, result, instruments = self.query_lstm(sequence)
        embedding_key = self.encode_key("embedding")
        points = {
            embedding_key: instruments["embedding"]
        }

        for part in NeuralNetwork.LSTM_PARTS:
            for layer in range(NeuralNetwork.LAYERS):
                points[self.encode_key(part, layer)] = instruments[part][layer]

        point_reductions, point_colours, point_predictions = self.compute_point_abstractions(points)
        embedding_name = self.latex_name(len(sequence) - 1, "embedding")
        embedding = HiddenState(embedding_name, point_reductions[embedding_key], colour=point_colours[embedding_key], predictions=self.prediction_distribution(point_predictions[embedding_key]))
        units = self.make_lstm_units(len(sequence) - 1, point_reductions, point_colours, point_predictions)
        softmax_name = self.latex_name(len(sequence) - 1, "softmax")
        colour_fn = lambda output: self.output_colour(output if not self.has_pos() else (self.pos_mapping[output] if output in self.pos_mapping else "NN"))
        softmax = LabelDistribution(softmax_name, result.distribution, self.sort_key, NeuralNetwork.OUTPUT_WIDTH, colour_fn)
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
        hidden_state = HiddenState(name, reduction[key], min_max, colour[key], self.prediction_distribution(prediction[key]))
        # This is the full point.
        full_hidden_state = HiddenState(name, point, min_max, None, None)
        back_links = {}

        for i, endpoint in self.dimensionality_reduce_mapping(keyed_point, NeuralNetwork.HIDDEN_REDUCTION)[key].items():
            for j in range(endpoint[0], endpoint[1]):
                assert j not in back_links, "%d already in %s" % (j, back_links)
                back_links[j] = i

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

        return LabelDistribution(None, prediction, self.sort_key, colour_fn=lambda output: self.output_colour(output))

    def make_lstm_units(self, timestep, point_reductions, point_colours, point_predictions):
        units = {}

        for part in NeuralNetwork.LSTM_PARTS:
            units[part] = {}

            for layer in range(NeuralNetwork.LAYERS):
                if part.endswith("_gates"):
                    min_max = (0, 1)
                else:
                    min_max = (None, None)

                key = self.encode_key(part, layer)
                name = self.latex_name(timestep, part, layer)
                hidden_state = HiddenState(name, point_reductions[key], min_max, point_colours[key], self.prediction_distribution(point_predictions[key]))
                units[part][layer] = hidden_state

        return units

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


class QueryEngine:
    TOP = 25

    def __init__(self):
        self.thresholds = {}
        #self._background_setup = threading.Thread(target=self._setup)
        #self._background_setup.daemon = True
        #self._background_setup.start()

    def _setup(self):
        activation_data_file = os.path.join(RESUME_DIR, "activation_data.pickle")
        logging.debug("Waiting on activation data.")

        while not os.path.exists(activation_data_file + MARKER):
            time.sleep(1)

        logging.debug("Processing activation data for query engine.")
        self.units = {}
        self.sequence_units = {}

        for i, activation_point in enumerate(pickler.load(activation_data_file)):
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

