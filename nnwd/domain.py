
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
from ml import nlp
from ml import nn as ffnn
from nnwd import geometry
from nnwd import latex
from nnwd.models import Timestep, WeightExplain, WeightDetail, HiddenState, LabelDistribution, SequenceRollup, SequenceMatch, Estimate
from nnwd import pickler
from nnwd import rnn
from pytils.log import setup_logging, user_log
from pytils import adjutant, check


ActivationPoint = collections.namedtuple("ActivationPoint", ["sequence", "expectation", "prediction", "part", "layer", "index", "point"])
MatchPoint = collections.namedtuple("MatchPoint", ["distance", "word", "index", "prediction", "expectation"])
MINIMUM_OCCURRENCE_COUNT = 2


def create_sa(task_form, corpus_stream_fn, aargs):
    train_xys_path = os.path.join(aargs.save_dir, "xys.train")
    validation_xys_path = os.path.join(aargs.save_dir, "xys.validation")
    test_xys_path = os.path.join(aargs.save_dir, "xys.test")
    sentiments_path = os.path.join(aargs.save_dir, "sentiments")
    output_distribution_path = os.path.join(aargs.save_dir, "output-distribution")
    words_path = os.path.join(aargs.save_dir, "words")

    if os.path.exists(words_path):
        train_xys = [xy for xy in pickler.load(train_xys_path)]
        validation_xys = [xy for xy in pickler.load(validation_xys_path)]
        test_xys = [xy for xy in pickler.load(test_xys_path)]
        sentiments = set([sentiment for sentiment in pickler.load(sentiments_path)])
        words = set([word for word in pickler.load(words_path)])
    else:
        train_xys = []
        validation_xys = []
        test_xys = []
        sentiments = set()
        total = 0.0
        output_distribution = {}
        words = {}

        for triple in corpus_stream_fn():
            sentiment = get_sentiment(triple[2])
            sentiments.add(sentiment)
            xy = ([(word, None) for word in triple[1]], sentiment)

            if triple[0] == "train":
                for word in triple[1]:
                    if word not in words:
                        words[word] = 0

                    words[word] += 1

                train_xys += [xy]

                if sentiment not in output_distribution:
                    output_distribution[sentiment] = 0

                output_distribution[sentiment] += 1
                total += 1
            elif triple[0] == "dev":
                validation_xys += [xy]
            else:
                test_xys += [xy]

        pickler.dump(train_xys, train_xys_path)
        pickler.dump(validation_xys, validation_xys_path)
        pickler.dump(test_xys, test_xys_path)
        pickler.dump(sorted(sentiments, key=sentiment_sort_key), sentiments_path)
        pickler.dump([output_distribution], output_distribution_path)
        words = set([item[0] for item in words.items() if item[1] > 1])
        pickler.dump([word for word in words], words_path)

    #printer = lambda sequences: "\n".join([" ".join([str(word_pos) for word_pos in i]) for i in sequences])
    #print(printer(train_xys))
    #print(printer(validation_xys))
    #print(printer(test_xys))
    logging.debug("data sets (train, validation, test): %d, %d, %d" % (len(train_xys), len(validation_xys), len(test_xys)))
    logging.debug("total pairs (t, v, t): %d, %d, %d" % (sum([len(xy[0]) for xy in train_xys]), sum([len(xy[0]) for xy in validation_xys]), sum([len(xy[0]) for xy in test_xys])))
    words = mlbase.Labels(words.union(set([mlbase.BLANK])), unknown=nlp.UNKNOWN)
    #print(words)
    output_labels = mlbase.Labels(sentiments)
    return words, NeuralNetwork(task_form, words, output_labels, None, None, train_xys, validation_xys, test_xys, lambda item: sentiment_sort_key(item[0]), aargs)


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


def create_lm(task_form, corpus_stream_fn, aargs):
    train_xys_path = os.path.join(aargs.save_dir, "xys.train")
    validation_xys_path = os.path.join(aargs.save_dir, "xys.validation")
    test_xys_path = os.path.join(aargs.save_dir, "xys.test")
    pos_tags_path = os.path.join(aargs.save_dir, "pos-tags")
    output_distribution_path = os.path.join(aargs.save_dir, "output-distribution")
    pos_mapping_path = os.path.join(aargs.save_dir, "pos-mapping")
    words_path = os.path.join(aargs.save_dir, "words")

    if os.path.exists(words_path):
        train_xys = [xy for xy in pickler.load(train_xys_path)]
        validation_xys = [xy for xy in pickler.load(validation_xys_path)]
        test_xys = [xy for xy in pickler.load(test_xys_path)]
        pos_tags = set([word for word in pickler.load(pos_tags_path, True)])
        pos_mapping = {item[0]: item[1] for item in pickler.load(pos_mapping_path, True)}
        words = set([word for word in pickler.load(words_path)])
    else:
        xys = [sentence for sentence in corpus_stream_fn()]
        random.shuffle(xys)
        split_1 = int(len(xys) * 0.8)
        split_2 = split_1 + int(len(xys) * 0.1)
        train_xys = xys[:split_1]
        validation_xys = xys[split_1:split_2]
        test_xys = xys[split_2:]
        ## Words are actually only the subset from the training data.
        #words = set(adjutant.flat_map([[word_pos[0] for word_pos in sentence] for sentence in train_xys]))
        pos_tags = set(adjutant.flat_map([[word_pos[1] for word_pos in sentence] for sentence in train_xys]))
        pickler.dump(train_xys, train_xys_path)
        pickler.dump(validation_xys, validation_xys_path)
        pickler.dump(test_xys, test_xys_path)
        pickler.dump([pos for pos in pos_tags], pos_tags_path)
        word_pos_counts = {}

        for sentence in train_xys:
            for word_pos in sentence:
                if word_pos[0] not in word_pos_counts:
                    word_pos_counts[word_pos[0]] = {}

                if word_pos[1] not in word_pos_counts[word_pos[0]]:
                    word_pos_counts[word_pos[0]][word_pos[1]] = 0

                word_pos_counts[word_pos[0]][word_pos[1]] += 1

        total = 0.0
        word_pos_counts2 = {}

        for word, counts in word_pos_counts.items():
            summed = sum(counts.values())

            if summed > MINIMUM_OCCURRENCE_COUNT:
                word_pos_counts2[word] = counts
                total += summed

        output_distribution = {}
        pos_mapping = {}

        for word, counts in word_pos_counts2.items():
            output_distribution[word] = sum(counts.values()) / total
            pos_count = sorted(counts.items(), key=lambda item: item[1], reverse=True)[0]
            pos_mapping[word] = pos_count[0]

        pickler.dump([output_distribution], output_distribution_path)
        pickler.dump([item for item in pos_mapping.items()], pos_mapping_path)
        words = set([word for word in word_pos_counts2.keys()])
        pickler.dump([word for word in words], words_path)

    #printer = lambda sequences: "\n".join([" ".join([str(word_pos) for word_pos in i]) for i in sequences])
    #print(printer(train_xys))
    #print(printer(validation_xys))
    #print(printer(test_xys))
    logging.debug("data sets (train, validation, test): %d, %d, %d" % (len(train_xys), len(validation_xys), len(test_xys)))
    logging.debug("total pairs (t, v, t): %d, %d, %d" % (sum([len(xy) for xy in train_xys]), sum([len(xy) for xy in validation_xys]), sum([len(xy) for xy in test_xys])))
    #print(words)
    word_labels = mlbase.Labels(words.union(set([mlbase.BLANK])), unknown=nlp.UNKNOWN)
    return word_labels, NeuralNetwork(task_form, word_labels, None, pos_tags, pos_mapping, xy_sequence(train_xys), xy_sequence(validation_xys), xy_sequence(test_xys), lambda item: -item[1], aargs)


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
    HIDDEN_WIDTH = 100
    EMBEDDING_WIDTH = 50
    OUTPUT_WIDTH = 5
    HIDDEN_REDUCTION = 10
    EMBEDDING_REDUCTION = 10
    MAXIMUM_CONSECUTIVE_DECAYS = 5
    TOP_PREDICTIONS = 3
    PREDICTOR_EPOCHS = 100
    PREDICTOR_SAMPLE_RATE = 0.5
    GAUSSIAN_SAMPLE_RATE = 0.5
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

    def __init__(self, task_form, words, outputs, pos_tags, pos_mapping, train_xys, validation_xys, test_xys, sort_key, aargs):
        self.task_form = task_form
        self.words = check.check_instance(words, mlbase.Labels)
        self.outputs = outputs
        self.pos_tags = pos_tags
        self.pos_mapping = pos_mapping
        self.train_xys = [mlbase.Xy(*pair) for pair in train_xys]
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
        self.batch = aargs.batch
        self.arc_epochs = aargs.arc_epochs
        self.save_dir = aargs.save_dir
        self.headless = aargs.headless
        self.skip_dr_test = aargs.skip_dr_test
        self.skip_pse_test = aargs.skip_pse_test
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
        # Sets setup_complete
        self._train_features()

        #if not self.skip_dr_test or not self.skip_sem_test:
        #    self._test_features()

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

        lstm_dir = os.path.join(self.save_dir, "lstm")

        if os.path.exists(lstm_dir):
            logging.debug("Loading existing lstm parameters.")
            self.lstm.load(lstm_dir)
        else:
            logging.debug("Training lstm parameters.")
            best_score_train = self.lstm.test(self.train_xys)
            best_score_validation = self.lstm.test(self.validation_xys)
            score_test = self.lstm.test(self.test_xys)
            logging.debug("Baseline train/validation/test scores (random initialized weights): %.4f / %.4f / %.4f" % (best_score_train, best_score_validation, score_test))
            training_parameters = mlbase.TrainingParameters() \
                .batch(self.batch) \
                .epochs(self.arc_epochs) \
                .convergence(False) \
                .debug(True) \
                .score(True)
            previous_loss = None
            arc = -1
            version = 0
            self.lstm.save(lstm_dir, version, True)
            converged = False
            consecutive_decays = 0

            while not converged:
                arc += 1
                logging.debug("train lstm arc %d: %s" % (arc, training_parameters))
                loss, score_train, score_validation = self.lstm_train_loop(training_parameters)
                loss_change = self._change(previous_loss, loss, lambda prev, curr: prev > curr)
                train_change = self._change(best_score_train, score_train, lambda prev, curr: prev < curr)
                validation_change = self._change(best_score_validation, score_validation, lambda prev, curr: prev < curr)
                logging.debug("train lstm arc %d: (loss, tr, va) (%s %.4f, %s %.4f, %s %.4f)" % (arc, loss_change, loss, train_change, score_train, validation_change, score_validation))
                both_improved = score_train > best_score_train and score_validation > best_score_validation

                if score_train > best_score_train or score_validation > best_score_validation:
                    previous_loss = loss
                    version += 1
                    self.lstm.save(lstm_dir, version, True)

                    # At least one improved.
                    if score_train > best_score_train:
                        best_score_train = score_train

                    if score_validation > best_score_validation:
                        best_score_validation = score_validation
                    else:
                        # The validation score didn't improve.  Lets see where the test score is at.
                        score_test = self.lstm.test(self.test_xys)
                        logging.debug("test score: %.4f" % score_test)
                else:
                    # Neither improved.
                    # Load the best known version to continue training off of.
                    self.lstm.load(lstm_dir)

                if not both_improved:
                    if consecutive_decays > NeuralNetwork.MAXIMUM_CONSECUTIVE_DECAYS:
                        converged = True

                    logging.debug("decaying..")
                    training_parameters = training_parameters.decay()
                    consecutive_decays += 1
                else:
                    consecutive_decays = 0

            # Load which ever version was marked as the latest as the final trained lstm.
            self.lstm.load(lstm_dir)

        logging.debug("Calculating final scores.")
        score_train = self.lstm.test(self.train_xys, False)
        score_validation = self.lstm.test(self.validation_xys, False)
        del self.validation_xys
        score_test = self.lstm.test(self.test_xys, True)
        logging.debug("(tr, va, te): (%.4f, %.4f, %.4f)" % (score_train, score_validation, score_test))

    def _change(self, previous, current, better_fn):
        if previous is None:
            return "-"
        elif better_fn(previous, current):
            return "▲"
        else:
            return "▼"

    def lstm_train_loop(self, training_parameters):
        loss, score_train = self.lstm.train(self.train_xys, training_parameters)
        score_validation = self.lstm.test(self.validation_xys)
        return loss, score_train, score_validation

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

    def _train_features(self):
        part_labels = mlbase.Labels(set(NeuralNetwork.INSTRUMENTS))
        layer_labels = mlbase.Labels(set(range(NeuralNetwork.LAYERS)))
        hidden_vector = mlbase.VectorField(max(NeuralNetwork.HIDDEN_WIDTH, NeuralNetwork.EMBEDDING_WIDTH))
        predictor_input = mlbase.ConcatField([part_labels, layer_labels, hidden_vector])

        if self.is_lm():
            #if self.has_pos():
            #    predictor_output = mlbase.Labels(self.pos_tags)
            #else:
            predictor_output = mlbase.Labels(set(self.words.labels()), unknown=nlp.UNKNOWN)
        else:
            predictor_output = mlbase.Labels(set(self.outputs.labels()))

        hyper_parameters = ffnn.HyperParameters() \
            .layers(2) \
            .width(len(predictor_input))
        self.predictor = ffnn.Model("predictor", hyper_parameters, predictor_input, predictor_output)
        predictor_dir = os.path.join(self.save_dir, "predictor")
        gaussian_buckets_path = os.path.join(self.save_dir, "gaussian-buckets")
        fixed_buckets_path = os.path.join(self.save_dir, "fixed-buckets")
        predictor_train_xys = None
        predictor_test_xys = None

        if os.path.exists(gaussian_buckets_path):
            logging.debug("Loading existing reduction buckets.")
            self.gaussian_buckets = {item[0]: item[1] for item in pickler.load(gaussian_buckets_path)}
            self.fixed_buckets = {item[0]: item[1] for item in pickler.load(fixed_buckets_path)}
        else:
            predictor_train_xys, predictor_test_xys = self._get_predictor_data()
            self._train_gaussian_buckets(predictor_train_xys, predictor_test_xys)

        # Technically not complete yet, but with the buckets setup an the predictor instantiated we can start answering queries.
        self.setup_complete = True

        if not self.skip_pse_test:
            if os.path.exists(predictor_dir):
                logging.debug("Loading existing predictor parameters.")
                self.predictor.load(predictor_dir)
            else:
                if predictor_train_xys is None:
                    predictor_train_xys, predictor_test_xys = self._get_predictor_data()

                self._train_predictor(predictor_train_xys, predictor_test_xys)
                self.predictor.save(predictor_dir)

        del predictor_train_xys
        del predictor_test_xys
        del self.train_xys

    def _train_predictor(self, predictor_train_xys, predictor_test_xys):
        logging.debug("Training predictor parameters.")
        training_parameters = mlbase.TrainingParameters() \
            .epochs(NeuralNetwork.PREDICTOR_EPOCHS) \
            .batch(32)
        loss = self.predictor.train(predictor_train_xys, training_parameters)
        accuracy = self.predictor.test(predictor_test_xys)
        logging.debug("train predictor %s, %s" % (loss, accuracy))

    def _train_gaussian_buckets(self, predictor_train_xys, predictor_test_xys):
        logging.debug("Training reduction buckets.")
        train_data = {}

        for xy in predictor_train_xys:
            part, layer, point = xy.x
            key = self.encode_key(part, layer)

            if key not in train_data:
                train_data[key] = []

            train_data[key] += [point]

        test_data = {}

        for xy in predictor_test_xys:
            part, layer, point = xy.x
            key = self.encode_key(part, layer)

            if key not in test_data:
                test_data[key] = []

            test_data[key] += [point]

        gaussian_buckets = {}
        fixed_buckets = {}

        for key, points in train_data.items():
            if self.decode_key(key)[0] == "embedding":
                width = NeuralNetwork.EMBEDDING_WIDTH
                reduction = NeuralNetwork.EMBEDDING_REDUCTION
            else:
                width = NeuralNetwork.HIDDEN_WIDTH
                reduction = NeuralNetwork.HIDDEN_REDUCTION

            logging.debug("dr calc for %s (%d -> %d) with %d data points sampled to %d" % (key, width, reduction, len(points), int(len(points) * NeuralNetwork.GAUSSIAN_SAMPLE_RATE)))
            gaussian_buckets[key] = [[] for i in range(reduction)]
            fixed_buckets[key] = []
            fixed_size = math.ceil(float(width) / reduction)

            # These are already drawn from shuffled data.
            #            vvvvvv
            # Since the data points come from the predictor dataset, they may be padded.
            # Truncate accordingly                                                          vvvvvv
            X = np.array(points[:int(len(points) * NeuralNetwork.GAUSSIAN_SAMPLE_RATE)])[:, :width] \
                .transpose()
            gm = GaussianMixture(reduction)
            dimension_grouping = gm.fit_predict(X)
            fixed_group = []
            incrementing_group = -1

            for dimension, group in enumerate(dimension_grouping):
                # If its the start of the transition to building out a new grouping of dimensions, and
                # if its the case that the remaining dimensions can be reduced evenly into a bucket of size less than 1, then do so.
                if len(fixed_group) == 0 and fixed_size > 1 and len(fixed_buckets[key]) + ((width - dimension) / (fixed_size - 1)) == reduction:
                    fixed_size -= 1

                gaussian_buckets[key][group] += [dimension]
                fixed_group += [dimension]

                if len(fixed_group) == fixed_size:
                    fixed_buckets[key] += [fixed_group]
                    fixed_group = []

        self.gaussian_buckets = gaussian_buckets
        self.fixed_buckets = fixed_buckets
        gaussian_buckets_path = os.path.join(self.save_dir, "gaussian-buckets")
        fixed_buckets_path = os.path.join(self.save_dir, "fixed-buckets")
        pickler.dump([item for item in self.gaussian_buckets.items()], gaussian_buckets_path)
        pickler.dump([item for item in self.fixed_buckets.items()], fixed_buckets_path)
        dr_errors = {}
        fixed_errors = {}
        count = None

        for key, points in test_data.items():
            if count is None:
                count = len(points)

            for point in points:
                # Learned buckets
                _, errors = self.gaussian_dimensionality_reduce({key: point}, True)

                for key, error in errors.items():
                    if key not in dr_errors:
                        dr_errors[key] = 0.0

                    dr_errors[key] += error

                # Fixed buckets
                _, errors = self.fixed_dimensionality_reduce({key: point}, True)

                for key, error in errors.items():
                    if key not in fixed_errors:
                        fixed_errors[key] = 0.0

                    fixed_errors[key] += error

        with open(os.path.join(self.save_dir, "dr-analysis.csv"), "w") as fh:
            logging.debug("Dimensionality reduction test counts: %d." % count)
            writer = csv_writer(fh)
            writer.writerow(["technique", "key", "sum of squared error", "mean squared error", "mse normalized"])

            for key, error in dr_errors.items():
                dimensions = NeuralNetwork.EMBEDDING_WIDTH if self.decode_key(key)[0] == "embedding" else NeuralNetwork.HIDDEN_WIDTH
                writer.writerow(["dr", key, "%f" % error, "%f" % (error / count), "%f" % (error / (count * dimensions))])

            for key, error in fixed_errors.items():
                dimensions = NeuralNetwork.EMBEDDING_WIDTH if self.decode_key(key)[0] == "embedding" else NeuralNetwork.HIDDEN_WIDTH
                writer.writerow(["fixed", key, "%f" % error, "%f" % (error / count), "%f" % (error / (count * dimensions))])

        return gaussian_buckets, fixed_buckets

    def _test_features(self):
        logging.debug("Testing features.")

        if not self.skip_dr_test:
            dr_errors = {}
            fixed_errors = {}

        manual_iteration = 0

        if not self.skip_sem_test:
            distributions_count = 0
            distributions = queue.Queue()
            pickler.dump(distributions, os.path.join(self.save_dir, "sem-distributions_batch-%d" % manual_iteration))

        manual_batch = len(self.test_xys)
        #manual_batch = max(1, int(len(self.test_xys) / 2.0))
        manual_offset = manual_iteration * manual_batch
        data_tenth = max(1, int(manual_batch / 10.0))
        data_hundredth = max(1, int(manual_batch / 100.0))
        count = 0

        for j, xy in enumerate(self.test_xys[manual_offset:manual_offset + manual_batch]):
            if j % data_hundredth == 0 or j + 1 == manual_batch:
                logging.debug("resource usage: %d" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

            if j % data_tenth == 0 or j + 1 == manual_batch:
                logging.debug("%d%% through.." % int((j + 1) * 100 / manual_batch))

            stepwise_lstm = self.lstm.stepwise(handle_unknown=True)

            for i, word_pos in enumerate(xy.x):
                count += 1
                result, instruments = stepwise_lstm.step(word_pos[0], NeuralNetwork.INSTRUMENTS)

                if not self.skip_sem_test:
                    x = ("embedding", 0, tuple(instruments["embedding"]) + self.embedding_padding)
                    xs = [x]

                if not self.skip_dr_test:
                    points = {self.encode_key("embedding", 0): tuple(instruments["embedding"])}

                for part in NeuralNetwork.LSTM_PARTS:
                    for layer in range(NeuralNetwork.LAYERS):
                        if not self.skip_sem_test:
                            x = (part, layer, tuple(instruments[part][layer]) + self.hidden_padding)
                            xs += [x]

                        if not self.skip_dr_test:
                            points[self.encode_key(part, layer)] = tuple(instruments[part][layer])

                if not self.skip_sem_test:
                    for result in self.predictor.evaluate(xs):
                        distributions_count += 1
                        distributions.put(result.distribution)

                    del xs

                if not self.skip_dr_test:
                    # Learned buckets
                    dr_reductions, errors = self.gaussian_dimensionality_reduce(points, True)

                    for key, error in errors.items():
                        if key not in dr_errors:
                            dr_errors[key] = 0.0

                        dr_errors[key] += error

                    # Fixed buckets
                    fixed_reductions, errors = self.fixed_dimensionality_reduce(points, True)

                    for key, error in errors.items():
                        if key not in fixed_errors:
                            fixed_errors[key] = 0.0

                        fixed_errors[key] += error

        if not self.skip_dr_test:
            logging.debug("Dimensionality reduction test counts: %d." % count)

            with open(os.path.join(self.save_dir, "dr-analysis-%d.csv" % manual_iteration), "w") as fh:
                writer = csv_writer(fh)
                writer.writerow(["technique", "key", "sum of squared error", "mean squared error", "mse normalized"])

                for key, error in dr_errors.items():
                    dimensions = NeuralNetwork.EMBEDDING_WIDTH if self.decode_key(key)[0] == "embedding" else NeuralNetwork.HIDDEN_WIDTH
                    writer.writerow(["dr", key, "%f" % error, "%f" % (error / count), "%f" % (error / (count * dimensions))])

                for key, error in fixed_errors.items():
                    dimensions = NeuralNetwork.EMBEDDING_WIDTH if self.decode_key(key)[0] == "embedding" else NeuralNetwork.HIDDEN_WIDTH
                    writer.writerow(["fixed", key, "%f" % error, "%f" % (error / count), "%f" % (error / (count * dimensions))])

        if not self.skip_sem_test:
            # Signal that the queue is complete.
            distributions.put(None)
            logging.debug("Predictor test distributions: %d." % distributions_count)

    def _get_predictor_data(self):
        if self.is_lm():
            #if self.has_pos():
            #    prediction_fn = lambda y, i: y[i][1]
            #else:
            prediction_fn = lambda y, i: y[i][0]
        else:
            # This is a sentiment analysis sa task.
            prediction_fn = lambda y, i: y

        predictor_train_xys = []
        predictor_train_xys_path = os.path.join(self.save_dir, "predictor-xys.train")
        predictor_test_xys = []
        predictor_test_xys_path = os.path.join(self.save_dir, "predictor-xys.test")

        if os.path.exists(predictor_test_xys_path):
            logging.debug("Loading existing predictor data.")
            predictor_train_xys = [xy for xy in pickler.load(predictor_train_xys_path)]
            predictor_test_xys = [xy for xy in pickler.load(predictor_test_xys_path)]
        else:
            logging.debug("Producing predictor data.")
            predictor_train_xys = self._derive_predictor_data(self.train_xys, prediction_fn)
            pickler.dump(predictor_train_xys, predictor_train_xys_path)
            predictor_test_xys = self._derive_predictor_data(self.test_xys, prediction_fn)
            pickler.dump(predictor_test_xys, predictor_test_xys_path)

        logging.debug("Predictor data: %d, %d." % (len(predictor_train_xys), len(predictor_test_xys)))
        return predictor_train_xys, predictor_test_xys

    def _derive_predictor_data(self, xys, prediction_fn):
        data = xys[:int(len(xys) * NeuralNetwork.PREDICTOR_SAMPLE_RATE)]
        data_quarter = max(1, int(len(data) / 4.0))
        predictor_xys = []
        instances = 0

        for j, xy in enumerate(data):
            if j % data_quarter == 0 or j + 1 == len(data):
                logging.debug("%d%% through.." % int((j + 1) * 100 / len(data)))

            stepwise_lstm = self.lstm.stepwise(handle_unknown=True)

            for i, word_pos in enumerate(xy.x):
                instances += 1
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

        logging.debug("derived instances: %d" % instances)
        return predictor_xys

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
                    result, instruments = stepwise_lstm.step(word_pos[0], NeuralNetwork.INSTRUMENTS)
                    part = "embedding"
                    layer = 0
                    point = tuple(instruments[part])
                    activation_data += [ActivationPoint(sequence=sequence, expectation=prediction_fn(xy.y, i), prediction=result.prediction, part=part, layer=layer, index=i, point=point)]

                    for part in NeuralNetwork.LSTM_PARTS:
                        for layer in range(NeuralNetwork.LAYERS):
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
        if self.is_lm():
            resolved = self.words.decode(self.words.encode(output, True))
        else:
            resolved = self.outputs.decode(self.outputs.encode(output, True))

        return resolved if not self.has_pos() else (self.pos_mapping[resolved] if resolved in self.pos_mapping else "NN")

    def output_colour(self, output):
        if not self.is_setup():
            return None

        return self.colour_mapping[self.mapped_output(output)]

    def rgb(self, colour):
        return "rgb(%d, %d, %d)" % colour

    def compute_point_abstractions(self, points):
        reductions, errors = self.gaussian_dimensionality_reduce(points)

        if not self.is_setup():
            colours = {key: None for key, point in points.items()}
            predictions = {key: None for key, point in points.items()}
            return reductions, colours, predictions

        predictions = self.predict_distributions(points)
        colours = self.fit_colours(points, predictions)
        return reductions, colours, predictions

    def _dimensionality_reduce(self, points, bucket_mapping, calculate_error=False):
        reductions = {}
        errors = {}

        for key, point in points.items():
            reduction = []
            errors[key] = 0.0

            for bucket, dimensions in enumerate(bucket_mapping[key]):
                value = sum([point[dimension] for dimension in dimensions]) / len(dimensions)
                reduction += [value]

                if calculate_error:
                    # Actual hidden state value compared to the reduced value.
                    #                    v                  v
                    errors[key] += sum([(point[dimension] - value)**2 for dimension in dimensions])

            reductions[key] = reduction

        return reductions, errors

    def fixed_dimensionality_reduce(self, points, calculate_error=False):
        return self._dimensionality_reduce(points, self.fixed_buckets, calculate_error)

    def gaussian_dimensionality_reduce(self, points, calculate_error=False):
        return self._dimensionality_reduce(points, self.gaussian_buckets, calculate_error)

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
            output_colourings = {}

            for word, probability in predictions[key].items():
                output = self.mapped_output(word)
                colour = self.output_colour(word)
                output_colourings[word] = (colour, probability)

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
        embedding_key = self.encode_key("embedding")
        points = {
            embedding_key: instruments["embedding"]
        }

        for part in NeuralNetwork.LSTM_PARTS:
            for layer in range(NeuralNetwork.LAYERS):
                points[self.encode_key(part, layer)] = instruments[part][layer]

        point_reductions, point_colours, point_predictions = self.compute_point_abstractions(points)
        embedding_name = self.latex_name(len(sequence) - 1, "embedding")
        embedding_name_no_t = self.latex_name_no_t("embedding")
        embedding = HiddenState(embedding_name, embedding_name_no_t, point_reductions[embedding_key], colour=point_colours[embedding_key], predictions=self.prediction_distribution(point_predictions[embedding_key]))
        units = self.make_lstm_units(len(sequence) - 1, point_reductions, point_colours, point_predictions)
        softmax_name = self.latex_name(len(sequence) - 1, "softmax")
        softmax = LabelDistribution(softmax_name, result.distribution, self.sort_key, NeuralNetwork.OUTPUT_WIDTH, lambda output: self.rgb(self.output_colour(output)))
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

        for bucket, dimensions in enumerate(self.gaussian_buckets[key]):
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

        for part in NeuralNetwork.LSTM_PARTS:
            units[part] = {}

            for layer in range(NeuralNetwork.LAYERS):
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

