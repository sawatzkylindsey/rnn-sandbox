#!/usr/bin/python
# -*- coding: utf-8 -*-

import collections
import json
import logging
import math
import numpy as np
import os
import pdb
import random
import re
import string
import tensorflow as tf
tf.logging.set_verbosity(logging.WARN)
from tensorflow.python import debug as tf_debug

from ml import base as mlbase
from pytils import adjutant, base, check
from pytils.log import user_log


LSTM_INSTRUMENTS = [
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
LSTM_MATRICES = {
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
LSTM_SINGULARS = {
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


class Lstm:
    # There are really only 2 states we care about capturing from the scan for the computation graph, however
    # we track all the other intermediate gates/states for the weight instrumentation.
    SCAN_STATES = 10

    def __init__(self, hyper_parameters, ablations, word_labels, output_labels, scope="rnn", skeleton=False):
        self.hyper_parameters = hyper_parameters
        self.ablations = ablations
        self.word_labels = word_labels
        self.output_labels = output_labels
        self.scope = scope

        if not skeleton:
            self.computational_graph()

    def computational_graph(self):
        self._initials = {}
        self._instruments = {}
        self._training_id = None

        self.time_dimension = None
        self.batch_dimension = None

        # Notation:
        #   _p      placeholder
        #   _c      constant
        self.unrolled_inputs_p = self.placeholder("unrolled_inputs_p", [self.time_dimension, self.batch_dimension], tf.int32)
        self.initial_state_p = self.placeholder("initial_state_p", [Lstm.SCAN_STATES, self.hyper_parameters.layers, self.batch_dimension, self.hyper_parameters.width])
        self.learning_rate_p = self.placeholder("learning_rate_p", [1], tf.float32)
        self.clip_norm_p = self.placeholder("clip_norm_p", [1], tf.float32)
        self.dropout_keep_p = self.placeholder("dropout_keep_p", [1], tf.float32)

        self.max_time, self.batch_size = tf.unstack(tf.shape(self.unrolled_inputs_p))

        self.E = self.variable("E", [len(self.word_labels), self.hyper_parameters.embedding_width])

        self.R = self.variable("R", [self.hyper_parameters.layers, self.hyper_parameters.width * 2, self.hyper_parameters.width])
        self.R_bias = self.variable("R_bias", [self.hyper_parameters.layers, self.hyper_parameters.width], initial=0.0)
        tf.identity(tf.reshape(self.R_bias, [-1, self.hyper_parameters.width]), name="R_bias")
        self.F = self.variable("F", [self.hyper_parameters.layers, self.hyper_parameters.width * 2, self.hyper_parameters.width])
        self.F_bias = self.variable("F_bias", [self.hyper_parameters.layers, self.hyper_parameters.width], initial=1.0)
        tf.identity(tf.reshape(self.F_bias, [-1, self.hyper_parameters.width]), name="F_bias")
        self.O = self.variable("O", [self.hyper_parameters.layers, self.hyper_parameters.width * 2, self.hyper_parameters.width])
        self.O_bias = self.variable("O_bias", [self.hyper_parameters.layers, self.hyper_parameters.width], initial=0.0)
        tf.identity(tf.reshape(self.O_bias, [-1, self.hyper_parameters.width]), name="O_bias")

        #self.dummy = tf.constant([0.0] * self.hyper_parameters.width, dtype="float32")
        #assert_shape(self.dummy, [self.hyper_parameters.width])
        #self.batched_dummy = tf.transpose(tf.reshape(tf.tile(self.dummy, [self.batch_size]), [self.hyper_parameters.width, self.batch_size]))

        if self.ablations.srnn:
            self.H = self.variable("H", [self.hyper_parameters.layers, self.hyper_parameters.width, self.hyper_parameters.width])
        else:
            self.H = self.variable("H", [self.hyper_parameters.layers, self.hyper_parameters.width * 2, self.hyper_parameters.width])
            self.H_bias = self.variable("H_bias", [self.hyper_parameters.layers, self.hyper_parameters.width], initial=0.0)
            tf.identity(tf.reshape(self.H_bias, [-1, self.hyper_parameters.width]), name="H_bias")

        self.Y = self.variable("Y", [self.hyper_parameters.width, len(self.output_labels)])
        self.Y_bias = self.variable("Y_bias", [1, len(self.output_labels)], initial=0.0)
        tf.identity(tf.reshape(self.Y_bias, [len(self.output_labels)]), name="Y_bias")

        self.unrolled_embedded_inputs = tf.nn.embedding_lookup(self.E, self.unrolled_inputs_p)
        assert_shape(self.unrolled_embedded_inputs, [self.time_dimension, self.batch_dimension, self.hyper_parameters.embedding_width])
        tf.identity(tf.reshape(self.unrolled_embedded_inputs, [-1, self.hyper_parameters.embedding_width]), name="embedding")

        if self.hyper_parameters.embedding_width != self.hyper_parameters.width:
            self.EP = self.variable("EP", [self.hyper_parameters.embedding_width, self.hyper_parameters.width])
            self.unrolled_embedded_projected_inputs = tf.matmul(tf.reshape(self.unrolled_embedded_inputs, [-1, self.hyper_parameters.embedding_width]), self.EP)
        else:
            self.unrolled_embedded_projected_inputs = tf.reshape(self.unrolled_embedded_inputs, [-1, self.hyper_parameters.embedding_width])

        assert_shape(self.unrolled_embedded_projected_inputs, [self.combine_dimensions(), self.hyper_parameters.width])

        # Dropout per: RECURRENT NEURAL NETWORK REGULARIZATION (Zaremba, Sutskever, Vinyals 2015)
        def step_lstm(previous_state, current_input):
            # This is all the other stacks, which we don't actually need for the looping (just there for instrumentation).
            #                                v
            output_previous, cell_previous, *_ = tf.unstack(previous_state)
            x = self.dropout(current_input)
            remember_gate_stack = []
            forget_gate_stack = []
            output_gate_stack = []
            input_hat_stack = []
            remember_stack = []
            cell_previous_stack = []
            forget_stack = []
            cell_stack = []
            cell_hat_stack = []
            output_stack = []

            for l in range(self.hyper_parameters.layers):
                assert_shape(output_previous[l], [self.batch_dimension, self.hyper_parameters.width])
                assert_shape(cell_previous[l], [self.batch_dimension, self.hyper_parameters.width])
                assert_shape(x, [self.batch_dimension, self.hyper_parameters.width])
                remember_gate = tf.sigmoid(tf.matmul(tf.concat([output_previous[l], x], axis=-1), self.R[l]) + self.R_bias[l])
                remember_gate_stack.append(remember_gate)
                forget_gate = tf.sigmoid(tf.matmul(tf.concat([output_previous[l], x], axis=-1), self.F[l]) + self.F_bias[l])
                forget_gate_stack.append(forget_gate)
                output_gate = tf.sigmoid(tf.matmul(tf.concat([output_previous[l], x], axis=-1), self.O[l]) + self.O_bias[l])
                output_gate_stack.append(output_gate)

                if self.ablations.srnn:
                    input_hat = tf.matmul(x, self.H[l])
                else:
                    input_hat = tf.tanh(tf.matmul(tf.concat([output_previous[l], x], axis=-1), self.H[l]) + self.H_bias[l])

                input_hat_stack.append(input_hat)
                remember = input_hat * remember_gate
                remember_stack.append(remember)
                cell_previous_stack.append(cell_previous[l])
                forget = cell_previous[l] * forget_gate
                forget_stack.append(forget)
                cell = forget + remember
                assert_shape(cell, [self.batch_dimension, self.hyper_parameters.width])
                cell_stack.append(cell)
                cell_hat = tf.tanh(cell)
                cell_hat_stack.append(cell_hat)

                if self.ablations.out:
                    output = cell_hat
                else:
                    output = cell_hat * output_gate

                assert_shape(output, [self.batch_dimension, self.hyper_parameters.width])
                output_stack.append(output)
                x = self.dropout(output)

            return tf.stack([output_stack, cell_stack, remember_gate_stack, forget_gate_stack, output_gate_stack, input_hat_stack, remember_stack, cell_previous_stack, forget_stack, cell_hat_stack])

        scan_inputs = tf.reshape(self.unrolled_embedded_projected_inputs, [self.max_time, self.batch_size, self.hyper_parameters.width])
        assert_shape(scan_inputs, [self.time_dimension, self.batch_dimension, self.hyper_parameters.width])
        self.unrolled_states = tf.scan(step_lstm, scan_inputs, self.initial_state_p)
        assert_shape(self.unrolled_states, [self.time_dimension, Lstm.SCAN_STATES, self.hyper_parameters.layers, self.batch_dimension, self.hyper_parameters.width])

        # We can produce the 'w' term irrespective of the ablations (or lack thereof).
        # However notice, this will only have the intended sum of soft filters interpretation under the srnn + out conditions.
        REMEMBER_POSITION = 2
        FORGET_POSITION = 3
        def step_w(index):
            left = self.unrolled_states[index][REMEMBER_POSITION]
            assert_shape(left, [self.hyper_parameters.layers, self.batch_dimension, self.hyper_parameters.width])

            right = tf.reduce_prod(self.unrolled_states[index + 1:, FORGET_POSITION, :], 0)
            assert_shape(right, [self.hyper_parameters.layers, self.batch_dimension, self.hyper_parameters.width])

            result = left * right
            assert_shape(result, [self.hyper_parameters.layers, self.batch_dimension, self.hyper_parameters.width])
            return result

        self.ws = tf.map_fn(step_w, tf.range(0, self.max_time), dtype=tf.float32)
        assert_shape(self.ws, [self.time_dimension, self.hyper_parameters.layers, self.batch_dimension, self.hyper_parameters.width])
        tf.identity(tf.reshape(self.ws, [self.max_time, self.hyper_parameters.layers, self.hyper_parameters.width]), name="ws")

        # Grab the last timesteps' state layers out of the unrolled state layers ([x y z] in diagram).
        # Notice, each cell in the diagram represents 2 (+all the extra instrumentation) states (hidden, cell, ..instrumentation..).
        #
        # state layer L     z z  z
        # ..                y y  y
        # state layer 0     x x  x
        # time              0 .. T
        self.state = self.unrolled_states[-1]
        assert_shape(self.state, [Lstm.SCAN_STATES, self.hyper_parameters.layers, self.batch_dimension, self.hyper_parameters.width])
        output_states, cell_states, remember_gates, forget_gates, output_gates, input_hats, remembers, cell_previouses, forgets, cell_hats = tf.unstack(self.state)
        tf.identity(tf.reshape(output_states, [-1, self.hyper_parameters.width]), name="outputs")
        tf.identity(tf.reshape(cell_states, [-1, self.hyper_parameters.width]), name="cells")
        tf.identity(tf.reshape(remember_gates, [-1, self.hyper_parameters.width]), name="remember_gates")
        tf.identity(tf.reshape(forget_gates, [-1, self.hyper_parameters.width]), name="forget_gates")
        tf.identity(tf.reshape(output_gates, [-1, self.hyper_parameters.width]), name="output_gates")
        tf.identity(tf.reshape(input_hats, [-1, self.hyper_parameters.width]), name="input_hats")
        tf.identity(tf.reshape(remembers, [-1, self.hyper_parameters.width]), name="remembers")
        tf.identity(tf.reshape(cell_previouses, [-1, self.hyper_parameters.width]), name="cell_previouses")
        tf.identity(tf.reshape(forgets, [-1, self.hyper_parameters.width]), name="forgets")
        tf.identity(tf.reshape(cell_hats, [-1, self.hyper_parameters.width]), name="cell_hats")

        # Grab the last state layer across all timesteps from the unrolled state layers ([z z z] in diagram).
        # Notice, each cell in the diagram represents 2 (+all the extra instrumentation) states (hidden, cell, ..instrumentation..).
        #
        # state layer L     z z  z
        # ..                y y  y
        # state layer 0     x x  x
        # time              0 .. T
        self.final_state = self.unrolled_states[:, 0, -1]
        assert_shape(self.final_state, [self.time_dimension, self.batch_dimension, self.hyper_parameters.width])

        final_state_for_matmul = tf.reshape(self.final_state, [-1, self.hyper_parameters.width])
        assert_shape(final_state_for_matmul, [self.combine_dimensions(), self.hyper_parameters.width])

        self.output_logits = tf.matmul(self.dropout(final_state_for_matmul), self.Y) + self.Y_bias
        assert_shape(self.output_logits, [self.combine_dimensions(), len(self.output_labels)])

        self.unrolled_outputs = tf.reshape(self.output_logits, [self.max_time, self.batch_size, len(self.output_labels)])
        assert_shape(self.unrolled_outputs, [self.time_dimension, self.batch_dimension, len(self.output_labels)])

        self.output_distributions = tf.nn.softmax(self.unrolled_outputs)
        assert_shape(self.output_distributions, [self.time_dimension, self.batch_dimension, len(self.output_labels)])

        self.cost = self.computational_graph_cost()
        #self.updates = tf.train.AdamOptimizer().minimize(self.cost)

        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate_p[0])
        gradients = optimizer.compute_gradients(self.cost)
        gradients_clipped = [(tf.clip_by_norm(g, self.clip_norm_p[0]), var) for g, var in gradients if g is not None]
        self.updates = optimizer.apply_gradients(gradients_clipped)

        #trainable_variables = tf.trainable_variables()
        #gradients = tf.gradients(self.cost, trainable_variables)
        #gradients_clipped, _ = tf.clip_by_global_norm(gradients, self.clip_norm_p[0])
        #optimizer = tf.train.GradientDescentOptimizer(self.learning_rate_p[0])
        #self.updates = optimizer.apply_gradients(zip(gradients_clipped, trainable_variables))

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def placeholder(self, name, shape, dtype=tf.float32):
        return tf.placeholder(dtype, shape, name=name)

    def variable(self, name, shape, dtype=tf.float32, initial=None):
        with tf.variable_scope(self.scope):
            return tf.get_variable(name, shape=shape, dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer() if initial is None else tf.constant_initializer(initial))

    def initial_state(self, batch_length):
        if batch_length not in self._initials:
            self._initials[batch_length] = np.zeros([Lstm.SCAN_STATES, self.hyper_parameters.layers, batch_length, self.hyper_parameters.width], dtype="float32")

        return self._initials[batch_length]

    def dropout(self, tensor):
        return tf.nn.dropout(tensor, self.dropout_keep_p[0])

    def combine_dimensions(self):
        if self.time_dimension is None or self.batch_dimension is None:
            return None
        else:
            return self.time_dimension * self.batch_dimension

    def train(self, xy_sequences, training_parameters):
        check.check_instance(training_parameters, mlbase.TrainingParameters)

        if id(xy_sequences) != self._training_id:
            self._training_id = id(xy_sequences)
            # Sort the training sequences by their length to minimize padding (each batch will consist of roughly equal lengthed sequences).
            self.training_xys = sorted(xy_sequences, key=lambda xy: len(xy.x))

        slot_length = len(str(training_parameters.epochs())) - 1
        case_slot_length = len(str(len(xy_sequences)))
        epoch_template = "Epoch training {:%dd} (loss, perplexity): {:.6f}, {:.6f}" % slot_length + (" (score {:.6f})" if training_parameters.score() else "")
        epochs_tenth = max(1, int(training_parameters.epochs() / 10))
        losses = training_parameters.losses()
        finished = False
        epoch = -1

        while not finished:
            epoch += 1
            epoch_loss = 0
            epoch_score = 0
            # Start at a different offset for every epoch to help avoid overfitting.
            offset = random.randint(0, min(training_parameters.batch(), len(self.training_xys)) - 1)
            count = 0
            first = True

            while offset < len(self.training_xys):
                if first:
                    first = False
                    batch = self.training_xys[0:offset]
                else:
                    batch = self.training_xys[offset:offset + training_parameters.batch()]
                    offset += training_parameters.batch()

                # To account for when offset is randomly assigned 0
                if len(batch) > 0:
                    count += len(batch)
                    feed = self.get_training_feed(batch, training_parameters)
                    _, loss = self.session.run([self.updates, self.cost], feed_dict=feed)
                    #_, loss, logits, targets = self.session.run([self.updates, self.cost, self.logits, self.targets], feed_dict=feed)
                    #_, loss, mask, uop1, lrs, mmm, mmn, mnn = self.session.run([self.updates, self.cost, self.mask, self.unrolled_outputs_p, self.losses_reduced, self.masked, self.masked2, self.masked3], feed_dict=feed)
                    #_, loss, mask, uop1, tgs, lrs, mmm = self.session.run([self.updates, self.cost, self.mask, self.unrolled_outputs_p, self.targets, self.losses_reduced, self.masked], feed_dict=feed)
                    #if epoch == 0:
                        #print(mask)
                        #print(uop1)
                        #print(tgs)
                        #print(lrs)
                        #print(mmm)
                        #print(mmn)
                        #print(mnn)
                        #print(dd)
                    epoch_loss += loss

                    if training_parameters.score():
                        feed = self.get_testing_feed(batch)
                        time_distributions = self.session.run(self.output_distributions, feed_dict=feed)
                        epoch_score += self.score(batch, feed, time_distributions, False, case_slot_length)

            assert count == len(xy_sequences), "%d != %d" % (count, len(xy_sequences))
            epoch_loss /= count
            epoch_perplexity = math.exp(epoch_loss)
            epoch_score /= count
            losses.append(epoch_loss)
            finished, reason = training_parameters.finished(epoch, losses)

            if not finished and epoch % epochs_tenth == 0 and training_parameters.debug():
                if training_parameters.score():
                    logging.debug(epoch_template.format(epoch, epoch_loss, epoch_perplexity, epoch_score))
                else:
                    logging.debug(epoch_template.format(epoch, epoch_loss, epoch_perplexity))

        if training_parameters.score():
            logging.debug(epoch_template.format(epoch, epoch_loss, epoch_perplexity, epoch_score))
        else:
            logging.debug(epoch_template.format(epoch, epoch_loss, epoch_perplexity))

        #logging.debug("Training finished due to %s (%s)." % (reason, losses))
        return epoch_loss, -epoch_perplexity

    def test(self, xy_sequences, debug=False, score=False):
        assert len(xy_sequences) > 0
        training_parameters = mlbase.TrainingParameters() \
            .dropout_rate(0)
        total_loss = 0.0
        total_score = 0.0
        case_slot_length = len(str(len(xy_sequences)))
        offset = 0

        while offset < len(xy_sequences):
            batch = xy_sequences[offset:offset + 32]
            offset += 32
            feed = self.get_training_feed(batch, training_parameters)
            time_distributions, loss = self.session.run([self.output_distributions, self.cost], feed_dict=feed)
            total_loss += loss

            if score:
                total_score += self.score(batch, feed, time_distributions, debug, case_slot_length)

        if score:
            logging.debug("total score for %d instances: %f" % (len(xy_sequences), total_score / len(xy_sequences)))

        return -math.exp(total_loss / len(xy_sequences))

    def evaluate_sequence(self, xs, handle_unknown=False, instrument_names=[]):
        feed = self.get_testing_feed([mlbase.Xy([(x, None) for x in xs], [])])
        instruments = self.get_instruments(instrument_names)
        distributions, *instrument_values = self.session.run([self.output_distributions] + instruments, feed_dict=feed)
        assert len(distributions) == len(xs), "%d != %d" % (len(distributions), len(xs))
        assert len(distributions[-1]) == 1, "%d != 1" % (len(distributions[-1]))
        distribution = distributions[-1][0]
        result = Result(self.output_labels.vector_decode(distribution), self.output_labels.vector_decode_distribution(distribution), self.output_labels.encoding())
        return result, {name: instrument_values[i] for i, name in enumerate(instrument_names)}

    def evaluate(self, x, handle_unknown=False, state=None, instrument_names=[]):
        feed = {
            self.unrolled_inputs_p: [[self.word_labels.encode(x, handle_unknown)]],
            self.initial_state_p: state if state is not None else self.initial_state(1),
            self.dropout_keep_p: np.array([1.0]),
        }
        instruments = self.get_instruments(instrument_names)
        distributions, next_state, *instrument_values = self.session.run([self.output_distributions, self.state] + instruments, feed_dict=feed)
        assert len(distributions) == 1
        assert len(distributions[-1]) == 1
        distribution = distributions[-1][0]
        result = Result(self.output_labels.vector_decode(distribution), self.output_labels.vector_decode_distribution(distribution), self.output_labels.encoding())
        return result, next_state, {name: instrument_values[i] for i, name in enumerate(instrument_names)}

    def get_instruments(self, instrument_names):
        key = hash(tuple(instrument_names))

        if key not in self._instruments:
            self._instruments[key] = [self.session.graph.get_tensor_by_name("%s:0" % name) for name in instrument_names]

        return self._instruments[key]

    def probe(self, name, layer):
        try:
            result = self.session.run(self.session.graph.get_tensor_by_name("%s:0" % (name)))
        except KeyError as e:
            result = self.session.run(self.session.graph.get_tensor_by_name("%s/%s:0" % (self.scope, name)))

        if layer is None:
            return result
        else:
            return result[layer]

    def stepwise(self, name=None, handle_unknown=False):
        return Stepwise(self, name, handle_unknown)

    def embed(self, x):
        feed = {
            self.unrolled_inputs_p: [[self.word_labels.encode(x, True)]],
            self.initial_state_p: self.initial_state(1),
            self.dropout_keep_p: np.array([1.0]),
        }
        return self.session.run([self.session.graph.get_tensor_by_name("embedding:0")], feed_dict=feed)[0].tolist()

    def load_parameters(self, model_dir, version=None):
        checkpoints = mlbase.Checkpoints.load(model_dir)
        model_path = checkpoints.model_path(version)
        version_key = "latest" if version is None else checkpoints.version_key(version)
        logging.debug("Restoring model %s=%s." % (version_key, model_path))
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope))
        saver.restore(self.session, model_path)

    def save_parameters(self, model_dir, version, set_latest=False):
        if os.path.isfile(model_dir) or (model_dir.endswith("/") and os.path.isfile(os.path.dirname(model_dir))):
            raise ValueError("model_dir '%s' must not be a file." % model_dir)

        checkpoints = mlbase.Checkpoints.load(model_dir)

        if checkpoints is None:
            checkpoints = mlbase.Checkpoints(model_dir)

        os.makedirs(model_dir, exist_ok=True)
        logging.debug("Saving model at %s=%d (latest=%s)." % (checkpoints.version_key(version), checkpoints.next_step, set_latest))
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope))
        saver.save(self.session, checkpoints.model_path_prefix(), global_step=checkpoints.next_step)
        checkpoints.update_next(version, set_latest) \
            .save()

    def copy(self, model_dir, version, set_latest=False):
        checkpoints = mlbase.Checkpoints.load(model_dir)
        checkpoints.version_key(version)
        copy_version = "%s-%s" % (version, "".join([random.choice(string.ascii_lowercase) for i in range(6)]))
        logging.debug("Copying model %s as %s (latest=%s)." % (checkpoints.version_key(version), checkpoints.version_key(copy_version), set_latest))
        checkpoints.copy(version, copy_version, set_latest) \
            .save()

    def keys(self):
        keys = [self.encode_key("embedding")]

        for part in LSTM_PARTS:
            for layer in range(self.hyper_parameters.layers):
                keys += [self.encode_key(part, layer)]

        return keys

    def part_layers(self):
        parts = [("embedding", 0)]

        for part in LSTM_PARTS:
            for layer in range(self.hyper_parameters.layers):
                parts += [(part, layer)]

        return parts

    def is_embedding(self, key_or_part, layer=None):
        if layer is None:
            part_actual, layer_actual = self.decode_key(key_or_part)
        else:
            part_actual = key_or_part

        return part_actual == "embedding"

    def part_width(self, key):
        if key == self.encode_key("embedding"):
            return self.hyper_parameters.embedding_width
        else:
            return self.hyper_parameters.width

    def encode_key(self, part, layer=None):
        return "%s-%d" % (part, 0 if layer is None else layer)

    def decode_key(self, key):
        result = key.split("-")

        if len(result) == 2:
            return (result[0], int(result[1]))
        else:
            raise ValueError("invalid decode of '%s': %s" % (key, result))


class LstmLm(Lstm):
    def __init__(self, hyper_parameters, ablations, word_labels, skeleton):
        super(LstmLm, self).__init__(hyper_parameters, ablations, word_labels, word_labels, skeleton=skeleton)
        pass

    def computational_graph_cost(self):
        self.input_lengths_p = self.placeholder("input_lengths_p", [self.batch_dimension], tf.int32)
        self.unrolled_outputs_p = self.placeholder("unrolled_outputs_p", [self.time_dimension, self.batch_dimension], tf.int32)
        self.mask = tf.sequence_mask(self.input_lengths_p, dtype=tf.float32)
        self.losses_reduced = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.unrolled_outputs, labels=self.unrolled_outputs_p)
        self.masked = tf.multiply(self.losses_reduced, tf.transpose(self.mask))
        self.masked2 = tf.reduce_sum(self.masked, 0)
        self.masked3 = tf.divide(self.masked2, tf.cast(self.input_lengths_p, tf.float32))
        return tf.reduce_sum(self.masked3)

    def xcomputational_graph_cost(self):
        self.input_lengths_p = self.placeholder("input_lengths_p", [self.batch_dimension], tf.int32)
        self.unrolled_outputs_p = self.placeholder("unrolled_outputs_p", [self.time_dimension, self.batch_dimension], tf.int32)
        self.mask = tf.sequence_mask(self.input_lengths_p, dtype=tf.float32)
        self.targets = tf.one_hot(self.unrolled_outputs_p, len(self.output_labels))
        logits = self.output_distributions
        epsilon = 1e-7
        losses = -tf.multiply(self.targets, tf.log(logits + epsilon)) - tf.multiply((1 - self.targets), tf.log(1 - logits + epsilon))
        assert_shape(losses, [self.time_dimension, self.batch_dimension, len(self.output_labels)])
        self.losses_reduced = tf.reduce_sum(losses, 2)
        assert_shape(self.losses_reduced, [self.time_dimension, self.batch_dimension])
        self.masked = tf.multiply(self.losses_reduced, tf.transpose(self.mask))
        self.masked2 = tf.reduce_sum(self.masked, 0)
        self.masked3 = tf.divide(self.masked2, tf.cast(self.input_lengths_p, tf.float32))
        return tf.reduce_mean(self.masked3)

    def ycomputational_graph_cost(self):
        self.input_lengths_p = self.placeholder("input_lengths_p", [self.batch_dimension], tf.int32)
        self.unrolled_outputs_p = self.placeholder("unrolled_outputs_p", [self.time_dimension, self.batch_dimension], tf.int32)
        self.mask = tf.sequence_mask(self.input_lengths_p, dtype=tf.float32)
        logits = tf.reshape(self.output_logits, [self.batch_size, self.max_time, len(self.output_labels)])
        targets = tf.reshape(self.unrolled_outputs_p, [self.batch_size, self.max_time])
        return tf.contrib.seq2seq.sequence_loss(logits=logits, targets=targets, weights=self.mask, average_across_timesteps=True, average_across_batch=True)

    def get_training_feed(self, batch, training_parameters):
        data_x, data_y = mlbase.as_time_major(batch, True)
        input_labels = [[self.word_labels.encode(word_pos[0] if word_pos is not None else mlbase.BLANK, True) for word_pos in timespot] for timespot in data_x]
        input_lengths = [len(sequence.x) for sequence in batch]
        output_labels = [[self.word_labels.encode(word_pos[0] if word_pos is not None else mlbase.BLANK, True) for word_pos in timespot] for timespot in data_y]
        return {
            self.unrolled_inputs_p: input_labels,
            self.input_lengths_p: input_lengths,
            self.initial_state_p: self.initial_state(len(batch)),
            self.learning_rate_p: [training_parameters.learning_rate()],
            self.clip_norm_p: [training_parameters.clip_norm()],
            self.dropout_keep_p: [1.0 - training_parameters.dropout_rate()],
            self.unrolled_outputs_p: output_labels,
        }

    def get_testing_feed(self, batch):
        data_x, data_y = mlbase.as_time_major(batch, True)
        input_labels = [[self.word_labels.encode(word_pos[0] if word_pos is not None else mlbase.BLANK, True) for word_pos in timespot] for timespot in data_x]
        input_lengths = [len(sequence.x) for sequence in batch]
        return {
            self.unrolled_inputs_p: input_labels,
            self.input_lengths_p: input_lengths,
            self.initial_state_p: self.initial_state(len(batch)),
            self.dropout_keep_p: np.array([1.0]),
        }

    def score(self, batch, feed, time_distributions, debug, case_slot_length):
        case_template = "{{Case {:%dd}}}" % case_slot_length
        input_lengths = feed[self.input_lengths_p]
        total_perplexity = 0.0

        if debug:
            sequence = [[] for case in range(len(batch))]
            predictions = [[] for case in range(len(batch))]
            predictions_probabilities = [[] for case in range(len(batch))]
            expectations = [[] for case in range(len(batch))]
            expectations_probabilities = [[] for case in range(len(batch))]

        log_probabilities = [0.0 for case in range(len(batch))]

        for timestep, distributions in enumerate(time_distributions):
            for case, distribution in enumerate(distributions):
                if timestep < input_lengths[case]:
                    if debug:
                        sequence[case] += [batch[case].x[timestep][0]]

                        predicted = self.output_labels.vector_decode(distribution)
                        predicted_probability = self.output_labels.vector_decode_probability(distribution, predicted)
                        predictions[case] += [predicted]
                        predictions_probabilities[case] += [predicted_probability]

                    expected = self.output_labels.decode(self.output_labels.encode(batch[case].y[timestep][0], True))
                    expected_probability = self.output_labels.vector_decode_probability(distribution, expected)

                    if debug:
                        expectations[case] += [expected]
                        expectations_probabilities[case] += [expected_probability]

                    log_probabilities[case] += math.log2(expected_probability)

        for case, log_probability in enumerate(log_probabilities):
            perplexity = 2**(-(log_probability / input_lengths[case]))
            total_perplexity += perplexity

            if debug:
                float_points = 4
                string_lengths = []

                for timestep in range(input_lengths[case]):
                    maximum = float_points + 2

                    if len(sequence[case][timestep]) > maximum:
                        maximum = len(sequence[case][timestep])

                    if len(predictions[case][timestep]) > maximum:
                        maximum = len(predictions[case][timestep])

                    if len(expectations[case][timestep]) > maximum:
                        maximum = len(expectations[case][timestep])

                    string_lengths += [maximum]

                debug_template = " ".join(["{:%d.%ds}" % (l, l) for l in string_lengths])
                float_template = "{:.%df}" % float_points
                sequence_str = debug_template.format(*(sequence[case]))
                predicted_str = debug_template.format(*predictions[case])
                predicted_probability_str = debug_template.format(*[float_template.format(p) for p in predictions_probabilities[case]])
                expected_str = debug_template.format(*expectations[case])
                expected_probability_str = debug_template.format(*[float_template.format(p) for p in expectations_probabilities[case]])
                debug_str = "   Inputed: %s\n Predicted: %s\n            %s\n  Expected: %s\n            %s" % \
                    (sequence_str, predicted_str, predicted_probability_str, expected_str, expected_probability_str)
                logging.debug("%s perplexity %.4f.\n%s" % (case_template.format(case), perplexity, debug_str))

        # Since 'score' means that the higher is better, but with perplexity the lower is better, so negate it.
        return -total_perplexity


class LstmSa(Lstm):
    def __init__(self, hyper_parameters, ablations, word_labels, output_labels, skeleton):
        super(LstmSa, self).__init__(hyper_parameters, ablations, word_labels, output_labels, skeleton=skeleton)
        pass

    def computational_graph_cost(self):
        self.input_gathers_p = self.placeholder("input_gathers_p", [self.batch_dimension, 2], tf.int32)
        self.output_p = self.placeholder("output_p", [self.batch_dimension], tf.int32)
        expected_outputs = tf.gather_nd(self.unrolled_outputs, self.input_gathers_p)
        return tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=expected_outputs, labels=self.output_p))

    def get_training_feed(self, batch, training_parameters):
        data_x, data_y = mlbase.as_time_major(batch, False)
        input_labels = [[self.word_labels.encode(word_pos[0] if word_pos is not None else mlbase.BLANK, True) for word_pos in timespot] for timespot in data_x]
        # Gathers are indexes, not lengths.
        #                                 vvv
        input_gathers = [[len(sequence.x) - 1, i] for i, sequence in enumerate(batch)]
        output_labels = [self.output_labels.encode(word if word is not None else mlbase.BLANK, True) for word in data_y]
        return {
            self.unrolled_inputs_p: input_labels,
            self.input_gathers_p: input_gathers,
            self.initial_state_p: self.initial_state(len(batch)),
            self.learning_rate_p: [training_parameters.learning_rate()],
            self.clip_norm_p: [training_parameters.clip_norm()],
            self.dropout_keep_p: [1.0 - training_parameters.dropout_rate()],
            self.output_p: output_labels,
        }

    def get_testing_feed(self, batch):
        data_x, data_y = mlbase.as_time_major(batch, False)
        input_labels = [[self.word_labels.encode(word_pos[0] if word_pos is not None else mlbase.BLANK, True) for word_pos in timespot] for timespot in data_x]
        # Gathers are indexes, not lengths.
        #                                 vvv
        input_gathers = [[len(sequence.x) - 1, i] for i, sequence in enumerate(batch)]
        return {
            self.unrolled_inputs_p: input_labels,
            self.input_gathers_p: input_gathers,
            self.initial_state_p: self.initial_state(len(batch)),
            self.dropout_keep_p: np.array([1.0]),
        }

    def score(self, batch, feed, time_distributions, debug, case_slot_length):
        case_template = "{{Case {:%dd}}}" % case_slot_length
        input_gathers = feed[self.input_gathers_p]
        total_correct = 0
        predictions = [[] for case in range(len(batch))]

        for timestep, distributions in enumerate(time_distributions):
            for case, distribution in enumerate(distributions):
                if timestep == input_gathers[case][0]:
                    prediction = self.output_labels.vector_decode(distribution)

                    if prediction == batch[case].y:
                        total_correct += 1

                        if debug:
                            logging.debug("%s passed!\n   Sequence: %s\n   Expected: %s" % \
                                (case_template.format(case), " ".join([word_pos[0] for word_pos in batch[case].x]), batch[case].y))
                    elif debug:
                        logging.debug("%s failed!\n   Sequence: %s\n   Expected: %s\n  Predicted: %s" % \
                            (case_template.format(case), " ".join([word_pos[0] for word_pos in batch[case].x]), batch[case].y, prediction))

        return total_correct


class Stepwise:
    def __init__(self, rnn, name=None, handle_unknown=False, state_t=None):
        self.rnn = rnn
        # Amazingly, python on our servers doesn't have random.choices.
        # We can do the same thing manually.              vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        self.name = name if name is not None else "".join([random.choice(string.ascii_lowercase) for i in range(6)])
        self.handle_unknown = handle_unknown
        self.state = None if state_t is None else state_t[0]
        self.t = 0 if state_t is None else state_t[1]

    def step(self, x, instrument_names=[]):
        result, self.state, instruments = self._query(x, instrument_names)
        self.t += 1
        return result, instruments

    def next_stepwise(self, x):
        _, next_state, _ = self._query(x)
        return Stepwise(self.rnn, self.name + "," + x, self.handle_unknown, (next_state, self.t + 1))

    def _query(self, x, instrument_names=[]):
        return self.rnn.evaluate(x, self.handle_unknown, self.state, instrument_names)

    def query(self, x, instrument_names=[]):
        result, state, instruments = self._query(x, instrument_names)
        return (result, instruments)


class Result:
    def __init__(self, prediction, distribution, encoding):
        self.prediction = prediction
        self.distribution = check.check_pdist(distribution)
        self.encoding = encoding

    def __repr__(self):
        return "(prediction=%s, distribution=%s)" % (self.prediction, sorted(self.distribution.items()))


def assert_shape(tensor, expected):
    assert tensor.shape.as_list() == expected, "actual %s != expected %s" % (tensor.shape, expected)

