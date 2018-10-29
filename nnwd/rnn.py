#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import numpy as np
import os
import pdb
import random
import re
import string
import tensorflow as tf
tf.logging.set_verbosity(logging.WARN)

import nlp
from pytils import adjutant, base, check
from pytils.log import user_log


class Rnn:
    def __init__(self, layers, width, word_labels):
        self.layers = layers
        self.width = width
        self.word_labels = word_labels

        # Notation:
        #   _p      placeholder
        #   _c      constant

        self.unrolled_inputs_p = self.placeholder("unrolled_inputs_p", [None, 1, len(self.word_labels)])
        self.output_label_p = self.placeholder("output_label_p", [None, 1, len(self.word_labels)])

        self.initial_state_p = self.placeholder("initial_state_p", [self.layers, 1, self.width])
        self.initial_state_c = np.zeros([self.layers, 1, self.width], dtype="float32")

        self.E = self.variable("E", [len(self.word_labels), self.width])
        self.E_bias = self.variable("E_bias", [1, self.width], 0.)

        self.H = self.variable("H", [self.layers, self.width * 2, self.width])
        self.H_bias = self.variable("H_bias", [self.layers, 1, self.width], 0.)

        self.Y = self.variable("Y", [self.width, len(self.word_labels)])
        self.Y_bias = self.variable("Y_bias", [1, len(self.word_labels)], 0.)

        self.unrolled_embedded_inputs = tf.matmul(tf.reshape(self.unrolled_inputs_p, [-1, len(self.word_labels)]), self.E) + self.E_bias

        def step_plain(previous_state, current_input):
            h_previous = tf.unstack(previous_state)
            x = current_input
            assert_shape(x, [1, self.width])
            h_stack = []

            for l in range(self.layers):
                assert_shape(h_previous[l], [1, self.width])
                h = tf.tanh(tf.matmul(tf.concat([h_previous[l], x], axis=-1), self.H[l]) + self.H_bias[l])
                h_stack.append(h)
                x = h

            return tf.stack(h_stack)

        self.unrolled_states = tf.scan(step_plain, tf.reshape(self.unrolled_embedded_inputs, [-1, 1, self.width]), self.initial_state_p)
        assert_shape(self.unrolled_states, [None, self.layers, 1, self.width])

        # Grab the last state (multi-layered) out of the unrolled state layers.
        self.state = self.unrolled_states[-1]
        assert_shape(self.state, [self.layers, 1, self.width])

        # Grab the last state layer across all timesteps from the unrolled state layers ([z z z] in diagram).
        #
        # state layer L     z z  z
        # ..                y y  y
        # state layer 0     x x  x
        # time              0 .. T
        self.final_state = self.unrolled_states[:, -1]
        assert_shape(self.final_state, [None, 1, self.width])

        final_state_for_matmul = tf.reshape(self.final_state, [-1, self.width])
        assert_shape(final_state_for_matmul, [None, self.width])

        self.output_logit = tf.tanh(tf.matmul(final_state_for_matmul, self.Y) + self.Y_bias)
        assert_shape(self.output_logit, [None, len(self.word_labels)])

        expected_output = tf.reshape(self.output_label_p, [-1, len(self.word_labels)])
        assert self.output_logit.shape.as_list() == expected_output.shape.as_list(), "%s != %s" % (self.output_logit.shape.as_list(), expected_output.shape.as_list())

        self.output_distribution = tf.nn.softmax(self.output_logit[0])
        assert_shape(self.output_distribution, [len(self.word_labels)])

        loss_fn = tf.nn.softmax_cross_entropy_with_logits_v2

        # Expected output:                 v
        # Un-scaled prediction:                                                      v
        self.cost = tf.reduce_mean(loss_fn(labels=tf.stop_gradient(expected_output), logits=self.output_logit))
        self.updates = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(self.cost)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def placeholder(self, name, shape):
        return tf.placeholder(tf.float32, shape, name=name)

    def variable(self, name, shape, initial=None):
        #with tf.variable_scope(self.scope):
        return tf.get_variable(name, shape=shape,
            initializer=tf.contrib.layers.xavier_initializer() if initial is None else tf.constant_initializer(initial))

    def train(self, xy_sequences, epochs=10, debug=False):
        shuffled_xy_sequences = xy_sequences.copy()
        slot_length = len(str(epochs)) - 1
        epoch_template = "Epoch {:%dd}: {:f}" % slot_length
        final_loss = None

        for epoch in range(epochs):
            epoch_loss = 0
            # Shuffle the training set for every epoch.
            random.shuffle(shuffled_xy_sequences)

            for sequence in shuffled_xy_sequences:
                assert len(sequence) > 0
                input_labels = [np.array([self.word_labels.ook_encode(xy.x)]) for xy in sequence]
                output_labels = [np.array([self.word_labels.ook_encode(xy.y)]) for xy in sequence]
                parameters = {
                    self.unrolled_inputs_p: input_labels,
                    self.initial_state_p: self.initial_state_c,
                    self.output_label_p: output_labels,
                }
                _, total_cost = self.session.run([self.updates, self.cost], feed_dict=parameters)
                epoch_loss += (total_cost / len(input_labels))

            logging.debug(epoch_template.format(epoch, epoch_loss))

            # Every 10th epoch (epochs start at 0, which is why we add 1).
            if debug and (epoch + 1) % 10 == 0:
                # Run the training sequence and compare the Rnn's output with that of what is expected.
                self.test(xy_sequences)

            if epoch + 1 == epochs:
                final_loss = epoch_loss

        return final_loss

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
                result, state = self.evaluate(xy.x, state)
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

    def evaluate(self, x, state=None):
        parameters = {
            self.unrolled_inputs_p: np.array([np.array([self.word_labels.ook_encode(x, True)])]),
            self.initial_state_p: state if state is not None else self.initial_state_c
        }
        distribution, next_state = self.session.run([self.output_distribution, self.state], feed_dict=parameters)
        return Result(self.word_labels.ook_decode(distribution), self.word_labels.ook_decode_distribution(distribution)), next_state

    def stepwise(self, name):
        return StepwiseEvaluation(self, name)


class Stepwise:
    def __init__(self, rnn, name=None):
        self.rnn = rnn
        self.state = None
        self.name = name if name is not None else "".join(random.choices(string.ascii_lowercase, k=6))
        self.t = 0

    def step(self, x):
        result, self.state = self.rnn.evaluate(x, self.state)
        logging.debug("%s @%3d: %s." % (self.name, self.t, result))
        self.t += 1
        return result


class Xy:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "(x=%s, y=%s)" % (self.x, self.y)

    def expected(self):
        if isinstance(self.y, dict):
            return max(self.y.items(), key=lambda item: item[1])[0]
        else:
            return self.y


class Result:
    def __init__(self, prediction, distribution):
        self.prediction = prediction
        self.distribution = distribution

    def __repr__(self):
        return "(prediction=%s, distribution=%s)" % (self.prediction, sorted(self.distribution.items()))


def assert_shape(tensor, expected):
    assert tensor.shape.as_list() == expected, "actual %s != expected %s" % (tensor.shape, expected)

