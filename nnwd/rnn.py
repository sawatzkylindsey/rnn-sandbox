#!/usr/bin/python
# -*- coding: utf-8 -*-

import collections
import logging
import numpy as np
import os
import pdb
import random
import re
import string
import tensorflow as tf
tf.logging.set_verbosity(logging.WARN)
from tensorflow.python import debug as tf_debug

from pytils import adjutant, base, check
from pytils.log import user_log


class Rnn:
    # There are really only 2 states we care about capturing from the scan for the computation graph, however
    # we track all the other intermediate gates/states for the weight instrumentation.
    SCAN_STATES = 10

    def __init__(self, layers, width, embedding_width, word_labels, scope="rnn"):
        self.layers = layers
        self.width = width
        self.embedding_width = embedding_width
        self.word_labels = word_labels
        self.scope = scope

        time_dimension = None
        batch_dimension = None
        # Notation:
        #   _p      placeholder
        #   _c      constant

        self.unrolled_inputs_p = self.placeholder("unrolled_inputs_p", [time_dimension], tf.int32)
        self.unrolled_outputs_p = self.placeholder("unrolled_outputs_p", [time_dimension], tf.int32)

        self.initial_state_p = self.placeholder("initial_state_p", [Rnn.SCAN_STATES, self.layers, 1, self.width])
        self.initial_state_c = np.zeros([Rnn.SCAN_STATES, self.layers, 1, self.width], dtype="float32")

        self.E = self.variable("E", [len(self.word_labels), self.embedding_width])
        self.EP = self.variable("EP", [self.embedding_width, self.width])

        self.R = self.variable("R", [self.layers, self.width * 2, self.width])
        self.R_bias = self.variable("R_bias", [self.layers, self.width], initial=0.0)
        tf.identity(tf.reshape(self.R_bias, [-1, self.width]), name="R_bias")
        self.F = self.variable("F", [self.layers, self.width * 2, self.width])
        self.F_bias = self.variable("F_bias", [self.layers, self.width], initial=1.0)
        tf.identity(tf.reshape(self.F_bias, [-1, self.width]), name="F_bias")
        self.O = self.variable("O", [self.layers, self.width * 2, self.width])
        self.O_bias = self.variable("O_bias", [self.layers, self.width], initial=0.0)
        tf.identity(tf.reshape(self.O_bias, [-1, self.width]), name="O_bias")

        self.H = self.variable("H", [self.layers, self.width * 2, self.width])
        self.H_bias = self.variable("H_bias", [self.layers, self.width], initial=0.0)
        tf.identity(tf.reshape(self.H_bias, [-1, self.width]), name="H_bias")

        self.Y = self.variable("Y", [self.width, len(self.word_labels)])
        self.Y_bias = self.variable("Y_bias", [1, len(self.word_labels)], initial=0.0)
        tf.identity(tf.reshape(self.Y_bias, [len(self.word_labels)]), name="Y_bias")

        self.unrolled_embedded_inputs = tf.nn.embedding_lookup(self.E, self.unrolled_inputs_p)
        assert_shape(self.unrolled_embedded_inputs, [None, self.embedding_width])
        tf.identity(tf.reshape(self.unrolled_embedded_inputs, [self.embedding_width]), name="embedding")

        self.unrolled_embedded_projected_inputs = tf.matmul(self.unrolled_embedded_inputs, self.EP)
        assert_shape(self.unrolled_embedded_projected_inputs, [None, self.width])

        def step_lstm(previous_state, current_input):
            # This is all the other stacks, which we don't actually need for the looping (just there for instrumentation).
            #                                v
            output_previous, cell_previous, *_ = tf.unstack(previous_state)
            x = current_input
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

            for l in range(self.layers):
                assert_shape(output_previous[l], [1, self.width])
                assert_shape(cell_previous[l], [1, self.width])
                assert_shape(x, [1, self.width])
                remember_gate = tf.sigmoid(tf.matmul(tf.concat([output_previous[l], x], axis=-1), self.R[l]) + self.R_bias[l])
                remember_gate_stack.append(remember_gate)
                forget_gate = tf.sigmoid(tf.matmul(tf.concat([output_previous[l], x], axis=-1), self.F[l]) + self.F_bias[l])
                forget_gate_stack.append(forget_gate)
                output_gate = tf.sigmoid(tf.matmul(tf.concat([output_previous[l], x], axis=-1), self.O[l]) + self.O_bias[l])
                output_gate_stack.append(output_gate)
                input_hat = tf.tanh(tf.matmul(tf.concat([output_previous[l], x], axis=-1), self.H[l]) + self.H_bias[l])
                input_hat_stack.append(input_hat)
                remember = input_hat * remember_gate
                remember_stack.append(remember)
                cell_previous_stack.append(cell_previous[l])
                forget = cell_previous[l] * forget_gate
                forget_stack.append(forget)
                cell = forget + remember
                assert_shape(cell, [1, self.width])
                cell_stack.append(cell)
                cell_hat = tf.tanh(cell)
                cell_hat_stack.append(cell_hat)
                output = cell_hat * output_gate
                assert_shape(output, [1, self.width])
                output_stack.append(output)
                x = output

            return tf.stack([output_stack, cell_stack, remember_gate_stack, forget_gate_stack, output_gate_stack, input_hat_stack, remember_stack, cell_previous_stack, forget_stack, cell_hat_stack])

        self.unrolled_states = tf.scan(step_lstm, tf.reshape(self.unrolled_embedded_projected_inputs, [-1, 1, self.width]), self.initial_state_p)
        assert_shape(self.unrolled_states, [None, Rnn.SCAN_STATES, self.layers, 1, self.width])

        # Grab the last timesteps' state layers out of the unrolled state layers ([x y z] in diagram).
        # Notice, each cell in the diagram represents 2 (+all the extra instrumentation) states (hidden, cell, ..instrumentation..).
        #
        # state layer L     z z  z
        # ..                y y  y
        # state layer 0     x x  x
        # time              0 .. T
        self.state = self.unrolled_states[-1]
        assert_shape(self.state, [Rnn.SCAN_STATES, self.layers, 1, self.width])
        output_states, cell_states, remember_gates, forget_gates, output_gates, input_hats, remembers, cell_previouses, forgets, cell_hats = tf.unstack(self.state)
        tf.identity(tf.reshape(output_states, [self.layers, self.width]), name="outputs")
        tf.identity(tf.reshape(cell_states, [self.layers, self.width]), name="cells")
        tf.identity(tf.reshape(remember_gates, [self.layers, self.width]), name="remember_gates")
        tf.identity(tf.reshape(forget_gates, [self.layers, self.width]), name="forget_gates")
        tf.identity(tf.reshape(output_gates, [self.layers, self.width]), name="output_gates")
        tf.identity(tf.reshape(input_hats, [self.layers, self.width]), name="input_hats")
        tf.identity(tf.reshape(remembers, [self.layers, self.width]), name="remembers")
        tf.identity(tf.reshape(cell_previouses, [self.layers, self.width]), name="cell_previouses")
        tf.identity(tf.reshape(forgets, [self.layers, self.width]), name="forgets")
        tf.identity(tf.reshape(cell_hats, [self.layers, self.width]), name="cell_hats")

        # Grab the last state layer across all timesteps from the unrolled state layers ([z z z] in diagram).
        # Notice, each cell in the diagram represents 2 (+all the extra instrumentation) states (hidden, cell, ..instrumentation..).
        #
        # state layer L     z z  z
        # ..                y y  y
        # state layer 0     x x  x
        # time              0 .. T
        self.final_state = self.unrolled_states[:, 0, -1]
        assert_shape(self.final_state, [None, 1, self.width])

        final_state_for_matmul = tf.reshape(self.final_state, [-1, self.width])
        assert_shape(final_state_for_matmul, [None, self.width])

        self.output_logit = tf.matmul(final_state_for_matmul, self.Y) + self.Y_bias
        assert_shape(self.output_logit, [None, len(self.word_labels)])

        self.output_distribution = tf.nn.softmax(self.output_logit[0])
        assert_shape(self.output_distribution, [len(self.word_labels)])

        loss_fn = tf.nn.softmax_cross_entropy_with_logits_v2

        # Expected output:                 v
        # Un-scaled prediction:                                                                                           v
        self.cost = tf.reduce_mean(loss_fn(labels=tf.stop_gradient(tf.one_hot(self.unrolled_outputs_p, len(self.word_labels))), logits=self.output_logit))
        self.updates = tf.train.AdamOptimizer().minimize(self.cost)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def placeholder(self, name, shape, dtype=tf.float32):
        return tf.placeholder(dtype, shape, name=name)

    def variable(self, name, shape, dtype=tf.float32, initial=None):
        with tf.variable_scope(self.scope):
            return tf.get_variable(name, shape=shape, dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer() if initial is None else tf.constant_initializer(initial))

    def train(self, xy_sequences, epoch_threshold=1000, loss_threshold=0.5, debug=False):
        shuffled_xy_sequences = xy_sequences.copy()
        slot_length = len(str(epoch_threshold)) - 1
        epoch_template = "Epoch training {:%dd}: {:f}" % slot_length
        epochs_tenth = int(epoch_threshold / 10)
        loss_window = 5
        losses = collections.deque([])
        finished = False
        epoch = -1

        while not finished:
            epoch += 1
            epoch_loss = 0
            # Shuffle the training set for every epoch.
            random.shuffle(shuffled_xy_sequences)

            for sequence in shuffled_xy_sequences:
                assert len(sequence) > 0

                if epoch == 0 and debug:
                    logging.debug("training on: %s" % sequence)

                input_labels = [self.word_labels.encode(xy.x) for xy in sequence]
                output_labels = [self.word_labels.encode(xy.y) for xy in sequence]
                parameters = {
                    self.unrolled_inputs_p: np.array(input_labels),
                    self.initial_state_p: self.initial_state_c,
                    self.unrolled_outputs_p: np.array(output_labels),
                }
                _, total_cost = self.session.run([self.updates, self.cost], feed_dict=parameters)
                epoch_loss += total_cost

            if epoch % epochs_tenth == 0:
                logging.debug(epoch_template.format(epoch, epoch_loss))

            losses.append(epoch_loss)

            if len(losses) > loss_window:
                losses.popleft()

            # Training is finished if:
            #   1. The current epoch exceeds the epochs threshold, or
            #   2. The losses are full and consistently lower than the loss threshold
            finished = epoch > epoch_threshold or (len(losses) == loss_window and all([loss < loss_threshold for loss in losses]))

        return epoch_loss

    def test(self, xy_sequences, debug=False):
        assert len(xy_sequences) > 0
        correct = 0
        total = 0
        case_slot_length = len(str(len(xy_sequences)))
        case_template = "{{Case {:%dd}}}" % case_slot_length

        for case, sequence in enumerate(xy_sequences):
            state = None
            distributions = []
            predictions = []
            test_pass = True

            for xy in sequence:
                result, state, _ = self.evaluate(xy.x, state)
                distributions.append(result.distribution)
                predictions.append(result.prediction)

                if result.prediction != xy.expected():
                    test_pass = False
                    break

            if test_pass:
                correct += 1

                if debug:
                    logging.debug("%s passed!\n  Full correctly predicted output: '%s'." % (case_template.format(case), predictions))
            else:
                suffix = " ..." if len(sequence) > len(predictions) else ""
                logging.debug("%s failed!\n  Expected: %s%s\n  Predicted: %s" % \
                    (case_template.format(case), " ".join([str(xy.expected()) for xy in sequence[:len(predictions)]]), suffix, " ".join([str(p) for p in predictions])))

            if debug:
                step_slot_length = len(str(len(distributions))) - 1
                output_template = "{:s} probability distribution at step {:%dd}: {:s}" % step_slot_length

                for t, distribution in enumerate(distributions):
                    logging.debug(output_template.format(case_template.format(case), t, adjutant.dict_as_str(distribution, False, True)))

            total += 1

        return correct / float(total)

    def evaluate(self, x, state=None, instrument_names=[]):
        parameters = {
            self.unrolled_inputs_p: np.array([self.word_labels.encode(x, True)]),
            self.initial_state_p: state if state is not None else self.initial_state_c
        }
        instruments = [self.session.graph.get_tensor_by_name("%s:0" % name) for name in instrument_names]
        distribution, next_state, *instrument_values = self.session.run([self.output_distribution, self.state] + instruments, feed_dict=parameters)
        result = Result(self.word_labels.ook_decode(distribution), self.word_labels.ook_decode_distribution(distribution), self.word_labels.encoding())
        return result, next_state, {name: instrument_values[i] for i, name in enumerate(instrument_names)}

    def probe(self, name, layer):
        try:
            result = self.session.run(self.session.graph.get_tensor_by_name("%s:0" % (name)))
        except KeyError as e:
            result = self.session.run(self.session.graph.get_tensor_by_name("%s/%s:0" % (self.scope, name)))

        if layer is None:
            return result
        else:
            return result[layer]

    def stepwise(self, name=None):
        return Stepwise(self, name)

    def embed(self, x):
        parameters = {
            self.unrolled_inputs_p: np.array([self.word_labels.encode(x, True)]),
            self.initial_state_p: self.initial_state_c
        }
        return self.session.run([self.session.graph.get_tensor_by_name("embedding:0")], feed_dict=parameters)[0].tolist()


class Stepwise:
    def __init__(self, rnn, name=None):
        self.rnn = rnn
        self.state = None
        self.name = name if name is not None else "".join(random.choices(string.ascii_lowercase, k=6))
        self.t = 0

    def step(self, x, instrument_names=[]):
        result, self.state, instruments = self.rnn.evaluate(x, self.state, instrument_names)
        logging.debug("%s @%3d: %s." % (self.name, self.t, result))
        self.t += 1
        return result, instruments


class Xy:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x \
            and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return "(x=%s, y=%s)" % (self.x, self.y)

    def expected(self):
        if isinstance(self.y, dict):
            return max(self.y.items(), key=lambda item: item[1])[0]
        else:
            return self.y


class Result:
    def __init__(self, prediction, distribution, encoding):
        self.prediction = prediction
        self.distribution = distribution
        self.encoding = encoding

    def __repr__(self):
        return "(prediction=%s, distribution=%s)" % (self.prediction, sorted(self.distribution.items()))


def assert_shape(tensor, expected):
    assert tensor.shape.as_list() == expected, "actual %s != expected %s" % (tensor.shape, expected)

