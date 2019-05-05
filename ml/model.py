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
from ml import scoring
from pytils import adjutant, check
from pytils.log import user_log


class Model:
    def __init__(self, scope, input_labels, output_labels):
        self.scope = scope
        self.input_labels = input_labels
        self.output_labels = output_labels

    def evaluate(self, x, handle_unknown=False):
        raise NotImplementedError()

    def _transform_x(self, x, handle_unknown):
        if isinstance(x, list):
            return [self.input_labels.vector_encode(i, handle_unknown) for i in x]
        else:
            return [self.input_labels.vector_encode(x, handle_unknown)]

    def test(self, xys, debug=False, score_fns={"accuracy": scoring.accuracy}):
        total_scores = {key: 0 for key in score_fns.keys()}
        count = 0
        case_slot_length = len(str(len(xys) if hasattr(xys, "__len__") else 1000000))
        case_template = "{{Case {:%dd}}}" % case_slot_length

        for case, xy in enumerate(xys):
            count += 1
            result = self.evaluate(xy.x)

            if debug:
                logging.debug("[%s] %s" % (self.scope, case_template.format(case)))

            for key, fn in score_fns.items():
                passed, score = fn(xy, result)
                total_scores[key] += score

                if debug:
                    if passed:
                        logging.debug("  Passed '%s' (%.4f)!\n  Full correctly predicted output: '%s'." % (key, score, result.prediction()))
                    else:
                        logging.debug("  Failed '%s' (%.4f)!\n  Expected: %s\n  Predicted: %s" % (key, score, str(xy.y), str(result.prediction())))

            #if debug:
            #    output_template = "[{:s}] {:s} probability distribution: {:s}"
            #    logging.debug(output_template.format(self.scope, case_template.format(case), adjutant.dict_as_str(result.distribution(), False, True, 6)))

        logging.info("Tested on %d instances." % count)
        # We count (rather then using len()) in case the xys come from a stream.
        #                          v
        return {key: score / float(count) for key, score in total_scores.items()}


class Ffnn(Model):
    def __init__(self, scope, input_labels, output_labels, hyper_parameters):
        super(Ffnn, self).__init__(scope, input_labels, output_labels)
        self.hyper = check.check_instance(hyper_parameters, HyperParameters)

        batch_size_dimension = None

        # Notation:
        #   _p      placeholder
        #   _c      constant

        # Base variable setup
        self.input_p = self.placeholder("input_p", [batch_size_dimension, len(self.input_labels)])
        self.output_p = self.placeholder("output_p", [batch_size_dimension], tf.int32)

        self.E = self.variable("E", [len(self.input_labels), self.hyper.width()])
        self.E_bias = self.variable("E_bias", [1, self.hyper.width()], 0.)

        # The E layer is the first layer.
        if self.hyper.layers() - 1 > 0:
            self.H = self.variable("H", [self.hyper.layers() - 1, self.hyper.width(), self.hyper.width()])
            self.H_bias = self.variable("H_bias", [self.hyper.layers() - 1, 1, self.hyper.width()], 0.)

        self.Y = self.variable("Y", [self.hyper.width(), len(self.output_labels)])
        self.Y_bias = self.variable("Y_bias", [1, len(self.output_labels)], 0.)

        # Computational graph encoding
        self.embedded_input = tf.tanh(tf.matmul(self.input_p, self.E) + self.E_bias)
        mlbase.assert_shape(self.embedded_input, [batch_size_dimension, self.hyper.width()])
        hidden = self.embedded_input
        mlbase.assert_shape(hidden, [batch_size_dimension, self.hyper.width()])

        for l in range(self.hyper.layers() - 1):
            hidden = tf.tanh(tf.matmul(hidden, self.H[l]) + self.H_bias[l])
            mlbase.assert_shape(hidden, [batch_size_dimension, self.hyper.width()])

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
                    xs = [self.input_labels.vector_encode(xy.x) for xy in batch]
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

    def evaluate(self, x, handle_unknown=False):
        xs = self._transform_x(x, handle_unknown)
        feed = {
            self.input_p: xs,
        }

        distributions = self.session.run(self.output_distributions, feed_dict=feed)

        if isinstance(x, list):
            return [mlbase.Result(self.output_labels, distribution) for distribution in distributions]
        else:
            return mlbase.Result(self.output_labels, distributions[0])

    def load(self, model_dir):
        model = tf.train.get_checkpoint_state(model_dir)
        assert model is not None, "No saved model in '%s'." % model_dir
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope))
        saver.restore(self.session, model.model_checkpoint_path)

    def save(self, model_dir):
        if os.path.isfile(model_dir) or (model_dir.endswith("/") and os.path.isfile(os.path.dirname(model_dir))):
            raise ValueError("model_dir '%s' must not be a file." % model_dir)

        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, "basename")
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope))
        saver.save(self.session, model_file)


class CustomOutput(Model):
    def __init__(self, scope, input_labels, output_labels, output_distribution):
        super(FixedModel, self).__init__(scope, input_labels, output_labels)
        check.check_pdist(output_distribution)
        assert len(output_labels) == len(output_distribution), "%d != %d" % (len(output_labels), len(output_distribution))

    def evaluate(self, x, handle_unknown=False):
        # Don't need xs, but run through the transformation to check data anyways.
        xs = self._transform_x(x, handle_unknown)

        if isinstance(x, list):
            return [mlbase.Result(self.output_labels, self.output_distribution) for i in range(len(x))]
        else:
            return mlbase.Result(self.output_labels, self.output_distribution)


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

        self._layers = check.check_gte(l, 1)
        return self

    def __repr__(self):
        return "HyperParameters{w=%d, l=%d}" % (self._width, self._layers)

