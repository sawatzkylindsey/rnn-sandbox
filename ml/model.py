#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
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
from ml import scoring
from pytils import adjutant, check
from pytils.log import user_log


LOSS = "loss"


class Model:
    def __init__(self, hyper_parameters, extra, input_field, output_labels, scope="model"):
        self.hyper_parameters = check.check_instance(hyper_parameters, HyperParameters)
        self.extra = extra
        self.input_field = input_field
        self.output_labels = output_labels
        self.scope = scope

    def evaluate(self, x, handle_unknown=False):
        raise NotImplementedError()

    def test(self, xys, debug=False, score_fns={}, include_loss=False):
        total_scores = {key: 0 for key in score_fns.keys()}

        if include_loss:
            total_scores[LOSS] = 0.0

        count = 0
        case_slot_length = len(str(len(xys) if hasattr(xys, "__len__") else 1000000))
        case_template = "{{Case {:%dd}}}" % case_slot_length
        batch = []
        cases = []
        total_loss = 0.0

        for case, xy in enumerate(xys):
            count += 1
            batch += [xy]
            cases += [case]

            if len(batch) == 100:
                for key, score in self._invoke(batch, cases, debug, score_fns, include_loss).items():
                    total_scores[key] += score

                batch = []
                cases = []

        if len(batch) > 0:
            for key, score in self._invoke(batch, cases, debug, score_fns, include_loss).items():
                total_scores[key] += score

        #logging.info("Tested on %d instances." % count)
        # We count (rather then using len()) in case the xys come from a stream.
        #                          v
        return {key: score / float(count) for key, score in total_scores.items()}

    def _invoke(self, batch, cases, debug, score_fns, include_loss):
        results, total_loss = self.evaluate(batch, True)
        scores = {key: 0 for key in score_fns.keys()}

        if include_loss:
            scores[LOSS] = total_loss

        if len(score_fns) > 0:
            for i, case in enumerate(cases):
                if debug:
                    logging.debug("[%s] %s" % (self.scope, case_template.format(case)))

                for key, fn in score_fns.items():
                    passed, score = fn(batch[i], results[i])
                    scores[key] += score

                    if debug:
                        if passed:
                            logging.debug("  Passed '%s' (%.4f)!\n  Full correctly predicted output: '%s'." % (key, score, results[i].prediction()))
                        else:
                            logging.debug("  Failed '%s' (%.4f)!\n  Expected: %s\n  Predicted: %s" % (key, score, str(results[i].y), str(results[i].prediction())))

        return scores


class Ffnn(Model):
    def __init__(self, hyper_parameters, extra, input_field, output_labels, scope):
        super(Ffnn, self).__init__(hyper_parameters, extra, input_field, output_labels, scope)

        batch_size_dimension = None

        # Notation:
        #   _p      placeholder
        #   _c      constant

        # Base variable setup
        self.input_p = self.placeholder("input_p", [batch_size_dimension, len(self.input_field)])
        self.output_p = self.placeholder("output_p", [batch_size_dimension], tf.int32)

        self.E = self.variable("E", [len(self.input_field), self.hyper_parameters.width])
        self.E_bias = self.variable("E_bias", [1, self.hyper_parameters.width], 0.)

        # The E layer is the first layer.
        if self.hyper_parameters.layers - 1 > 0:
            self.H = self.variable("H", [self.hyper_parameters.layers - 1, self.hyper_parameters.width, self.hyper_parameters.width])
            self.H_bias = self.variable("H_bias", [self.hyper_parameters.layers - 1, 1, self.hyper_parameters.width], 0.)

        self.Y = self.variable("Y", [self.hyper_parameters.width, len(self.output_labels)])
        self.Y_bias = self.variable("Y_bias", [1, len(self.output_labels)], 0.)

        # Computational graph encoding
        self.embedded_input = tf.tanh(tf.matmul(self.input_p, self.E) + self.E_bias)
        mlbase.assert_shape(self.embedded_input, [batch_size_dimension, self.hyper_parameters.width])
        hidden = self.embedded_input
        mlbase.assert_shape(hidden, [batch_size_dimension, self.hyper_parameters.width])

        for l in range(self.hyper_parameters.layers - 1):
            hidden = tf.tanh(tf.matmul(hidden, self.H[l]) + self.H_bias[l])
            mlbase.assert_shape(hidden, [batch_size_dimension, self.hyper_parameters.width])

        self.output_logit = tf.matmul(hidden, self.Y) + self.Y_bias
        mlbase.assert_shape(self.output_logit, [batch_size_dimension, len(self.output_labels)])
        self.output_distributions = tf.nn.softmax(self.output_logit)
        #self.output_distributions = tf.nn.softmax(tf.matmul(hidden, self.Y) + self.Y_bias)
        mlbase.assert_shape(self.output_distributions, [batch_size_dimension, len(self.output_labels)])
        #self.cost = tf.reduce_mean(tf.nn.nce_loss(
        #    weights=tf.transpose(self.Y),
        #    biases=self.Y_bias,
        #    labels=self.output_p,
        #    inputs=hidden,
        #    num_sampled=1,
        #    num_classes=len(self.output_labels)))
        loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits
        self.cost = tf.reduce_sum(loss_fn(labels=tf.stop_gradient(self.output_p), logits=self.output_logit))
        self.updates = tf.train.AdamOptimizer().minimize(self.cost)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def placeholder(self, name, shape, dtype=tf.float32):
        return tf.placeholder(dtype, shape, name=name)

    def variable(self, name, shape, initial=None):
        with tf.variable_scope(self.scope):
            return tf.get_variable(name, shape=shape,
                initializer=tf.contrib.layers.xavier_initializer() if initial is None else tf.constant_initializer(initial))

    def train(self, xys, training_parameters):
        check.check_instance(training_parameters, mlbase.TrainingParameters)
        shuffled_xys = check.check_iterable_of_instances(xys, mlbase.Xy).copy()
        # Shuffle the training set for every epoch.
        random.shuffle(shuffled_xys)
        slot_length = len(str(training_parameters.epochs())) - 1
        epoch_template = "[%s] Epoch {:%dd}: {:f}" % (self.scope, slot_length)
        final_loss = None
        epochs_tenth = max(1, int(training_parameters.epochs() / 10))
        losses = training_parameters.losses()
        finished = False
        epoch = -1

        while not finished:
            epoch += 1
            epoch_loss = 0
            # Start at a different offset for every epoch to help avoid overfitting.
            offset = random.randint(0, min(training_parameters.batch(), len(shuffled_xys)) - 1)
            first = True
            count = 0

            while offset < len(shuffled_xys):
                if first:
                    first = False
                    batch = shuffled_xys[0:offset]
                else:
                    batch = shuffled_xys[offset:offset + training_parameters.batch()]
                    offset += training_parameters.batch()

                # To account for when offset is randomly assigned 0
                if len(batch) > 0:
                    count += len(batch)
                    xs = [self.input_field.vector_encode(xy.x, True) for xy in batch]
                    ys = [self.output_labels.encode(xy.y, True) for xy in batch]
                    feed = {
                        self.input_p: xs,
                        self.output_p: ys,
                    }
                    _, training_loss = self.session.run([self.updates, self.cost], feed_dict=feed)
                    offset += training_parameters.batch()
                    epoch_loss += training_loss

            epoch_loss /= count
            losses.append(epoch_loss)
            finished, reason = training_parameters.finished(epoch, losses)

            if not finished and epoch % epochs_tenth == 0:
                logging.debug(epoch_template.format(epoch, epoch_loss))

                if training_parameters.debug():
                    # Run the training data and compare the network's output with that of what is expected.
                    self.test(xys)

        logging.debug(epoch_template.format(epoch, epoch_loss))
        logging.debug("Training on %d instances finished due to %s (%s)." % (len(shuffled_xys), reason, losses))
        return epoch_loss

    def evaluate(self, batch, handle_unknown=False):
        xs = [self.input_field.vector_encode(xy.x, handle_unknown) for xy in batch]
        ys = [self.output_labels.encode(xy.y, True) for xy in batch]
        feed = {
            self.input_p: xs,
            self.output_p: ys,
        }

        distributions, loss = self.session.run([self.output_distributions, self.cost], feed_dict=feed)

        if isinstance(batch, list):
            return [mlbase.Result(self.output_labels, distribution) for distribution in distributions], loss
        else:
            return mlbase.Result(self.output_labels, distributions[0]), loss

    def load_parameters(self, model_dir):
        model = tf.train.get_checkpoint_state(model_dir)
        assert model is not None, "No saved model in '%s'." % model_dir
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope))
        saver.restore(self.session, model.model_checkpoint_path)

    def save_parameters(self, model_dir):
        if os.path.isfile(model_dir) or (model_dir.endswith("/") and os.path.isfile(os.path.dirname(model_dir))):
            raise ValueError("model_dir '%s' must not be a file." % model_dir)

        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, "basename")
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope))
        saver.save(self.session, model_file)


class CustomOutput(Model):
    def __init__(self, hyper_parameters, extra, input_field, output_labels, scope, output_distribution):
        super(CustomOutput, self).__init__(hyper_parameters, extra, input_field, output_labels, scope)
        check.check_pdist(output_distribution)
        assert len(output_labels) == len(output_distribution), "%d != %d" % (len(output_labels), len(output_distribution))
        self.output_distribution = output_distribution

    def evaluate(self, batch, handle_unknown=False):
        # Don't need xs, but run through the transformation to check data anyways.
        xs = [self.input_field.vector_encode(xy.x, handle_unknown) for xy in batch]

        if isinstance(batch, list):
            return [mlbase.Result(self.output_labels, self.output_distribution) for i in range(len(batch))], None
        else:
            return mlbase.Result(self.output_labels, self.output_distribution), None


class HyperParameters:
    def __init__(self, layers, width):
        self.width = layers
        self.layers = width

    def __repr__(self):
        return "HyperParameters{l=%d, w=%d}" % (self.layers, self.width)

