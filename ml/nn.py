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

from ml import base as mlbase
from pytils import adjutant, check
from pytils.log import user_log



class Model:
    def __init__(self, scope, hyper_parameters, input_labels, output_labels, task):
        self.scope = scope
        self.hyper = check.check_instance(hyper_parameters, HyperParameters)
        self.input_labels = input_labels
        self.output_labels = output_labels
        self.task = task

        batch_size_dimension = None

        # Notation:
        #   _p      placeholder
        #   _c      constant

        # Base variable setup
        self.input_p = self.placeholder("input_p", [batch_size_dimension, len(self.input_labels)])
        self.output_p = self.placeholder("output_p", [batch_size_dimension, len(self.output_labels)])

        self.E = self.variable("E", [len(self.input_labels), self.hyper.width()])
        self.E_bias = self.variable("E_bias", [1, self.hyper.width()], 0.)

        if self.hyper.layers() > 0:
            self.H = self.variable("H", [self.hyper.layers(), self.hyper.width(), self.hyper.width()])
            self.H_bias = self.variable("H_bias", [self.hyper.layers(), 1, self.hyper.width()], 0.)

        self.Y = self.variable("Y", [self.hyper.width(), len(self.output_labels)])
        self.Y_bias = self.variable("Y_bias", [1, len(self.output_labels)], 0.)

        # Computational graph encoding
        self.embedded_input = tf.matmul(self.input_p, self.E) + self.E_bias
        mlbase.assert_shape(self.embedded_input, [batch_size_dimension, self.hyper.width()])
        hidden = self.embedded_input
        mlbase.assert_shape(hidden, [batch_size_dimension, self.hyper.width()])

        for l in range(self.hyper.layers()):
            hidden = tf.tanh(tf.matmul(hidden, self.H[l]) + self.H_bias[l])
            mlbase.assert_shape(hidden, [batch_size_dimension, self.hyper.width()])

        self.output_logit = tf.tanh(tf.matmul(hidden, self.Y) + self.Y_bias)
        mlbase.assert_shape(self.output_logit, [batch_size_dimension, len(self.output_labels)])
        self.output_distribution = tf.nn.softmax(self.output_logit[0])
        mlbase.assert_shape(self.output_distribution, [len(self.output_labels)])
        #expected_output = tf.reshape(self.output_label_p, [-1, len(self.output_labels)])
        #assert self.output_logit.shape.as_list() == expected_output.shape.as_list(), "%s != %s" % (self.output_logit.shape.as_list(), expected_output.shape.as_list())

        if self.task == mlbase.SINGLE_LABEL:
            loss_fn = tf.nn.softmax_cross_entropy_with_logits_v2
        elif self.task == mlbase.MULTI_LABEL:
            loss_fn = tf.nn.sigmoid_cross_entropy_with_logits

        # Expected output:                 v
        # Un-scaled prediction:                                                      v
        self.cost = tf.reduce_mean(loss_fn(labels=tf.stop_gradient(self.output_p), logits=self.output_logit))
        self.updates = tf.train.AdamOptimizer().minimize(self.cost)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def placeholder(self, name, shape):
        return tf.placeholder(tf.float32, shape, name=name)

    def variable(self, name, shape, initial=None):
        with tf.variable_scope(self.scope):
            return tf.get_variable(name, shape=shape,
                initializer=tf.contrib.layers.xavier_initializer() if initial is None else tf.constant_initializer(initial))

    def train(self, xys, training_parameters):
        check.check_instance(training_parameters, mlbase.TrainingParameters)
        shuffled_xys = check.check_iterable_of_instances(xys, mlbase.Xy).copy()
        slot_length = len(str(training_parameters.epochs())) - 1
        epoch_template = "[%s] Epoch {:%dd}: {:f}" % (self.scope, slot_length)
        final_loss = None
        epochs_tenth = int(training_parameters.epochs() / 10)
        losses = training_parameters.losses()
        finished = False
        epoch = -1

        while not finished:
            epoch += 1
            epoch_loss = 0
            # Shuffle the training set for every epoch.
            random.shuffle(shuffled_xys)
            offset = 0

            while offset < len(shuffled_xys):
                batch = shuffled_xys[offset:offset + training_parameters.batch()]
                xs = [self.input_labels.vector_encode(xy.x) for xy in batch]
                ys = [self.output_labels.vector_encode(xy.y) for xy in batch]
                feed = {
                    self.input_p: np.array(xs),
                    self.output_p: np.array(ys),
                }
                _, training_loss = self.session.run([self.updates, self.cost], feed_dict=feed)
                offset += training_parameters.batch()
                epoch_loss += training_loss

            losses.append(epoch_loss)

            if epoch % epochs_tenth == 0:
                logging.debug(epoch_template.format(epoch, epoch_loss))

                if training_parameters.debug():
                    # Run the training data and compare the network's output with that of what is expected.
                    self.test(xys)

            finished, reason = training_parameters.finished(epoch, losses)

        logging.debug("Training finished due to %s (%s)." % (reason, losses))
        logging.debug(epoch_template.format(epoch, epoch_loss))
        return epoch_loss

    def test(self, xys, debug=False):
        correct = 0
        total = 0
        case_slot_length = len(str(len(xys)))
        case_template = "{{Case {:%dd}}}" % case_slot_length

        for case, xy in enumerate(xys):
            test_pass = True
            result = self.evaluate(xy.x)

            if result.prediction != xy.y:
                test_pass = False

            if test_pass:
                correct += 1

                if debug:
                    logging.debug("[%s] %s passed!\n  Full correctly predicted output: '%s'." % (self.scope, case_template.format(case), result.prediction))
            else:
                logging.debug("[%s] %s failed!\n  Expected: %s\n  Predicted: %s" % \
                    (self.scope, case_template.format(case), str(xy.y), str(result.prediction)))

            if debug:
                output_template = "[{:s}] {:s} probability distribution: {:s}"
                logging.debug(output_template.format(self.scope, case_template.format(case), adjutant.dict_as_str(result.distribution, False, True)))

            total += 1

        return correct / float(total)

    def evaluate(self, x):
        feed = {
            self.input_p: np.array([self.input_labels.vector_encode(x, True)]),
        }
        distribution = self.session.run(self.output_distribution, feed_dict=feed)
        return mlbase.Result(self.output_labels.vector_decode(distribution), self.output_labels.vector_decode_distribution(distribution))

    def stepwise(self, name=None):
        return Stepwise(self, name)


class Stepwise:
    def __init__(self, nn, name=None):
        self.nn = nn
        self.name = name if name is not None else "".join(random.choices(string.ascii_lowercase, k=6))

    def step(self, x):
        result = self.nn.evaluate(x)
        logging.debug("%s: %s" % (self.name, result))
        return result


class HyperParameters:
    DEFAULT_WIDTH = 10
    DEFAULT_LAYERS = 1

    def __init__(self):
        self._width = HyperParameters.DEFAULT_WIDTH
        self._layers = HyperParameters.DEFAULT_LAYERS

    def width(self, w=None):
        if w is None:
            return self._width

        self._width = check.check_gte(w, 1)
        return self

    def layers(self, l=None):
        if l is None:
            return self._layers

        self._layers = check.check_gte(l, 0)
        return self

    def __repr__(self):
        return "HyperParameters{w=%d, l=%d}" % (self._width, self._layers)

