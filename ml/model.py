#!/usr/bin/python
# -*- coding: utf-8 -*-

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

from ml import base as mlbase
from ml import scoring
from pytils import adjutant, check
from pytils.log import user_log


LOSS = "loss"
PERPLEXITY = "perplexity"


class Model:
    def __init__(self, scope="model"):
        self.scope = scope

    def evaluate(self, x, handle_unknown=False):
        raise NotImplementedError()

    def test(self, xys_stream, debug=False):
        total_loss = 0.0
        count = 0
        case_slot_length = len(str(len(xys_stream) if hasattr(xys_stream, "__len__") else 1000000))
        case_template = "{{Case {:%dd}}}" % case_slot_length
        batch = []
        cases = []
        total_loss = 0.0

        for case, xy in enumerate(xys_stream()):
            count += 1
            batch += [xy]
            cases += [case]

            if len(batch) == 100:
                results, loss = self.evaluate(batch, True)
                total_loss += loss
                batch = []
                cases = []

        if len(batch) > 0:
            results, loss = self.evaluate(batch, True)
            total_loss += loss

        # We count (rather then using len()) in case the xys come from a stream.
        #                             v
        return -math.exp(total_loss / count)


class TfModel(Model):
    def __init__(self, scope):
        super(TfModel, self).__init__(scope)

    def placeholder(self, name, shape, dtype=tf.float32):
        return tf.placeholder(dtype, shape, name=name)

    def variable(self, name, shape, initial=None):
        with tf.variable_scope(self.scope):
            return tf.get_variable(name, shape=shape,
                initializer=tf.contrib.layers.xavier_initializer() if initial is None else tf.constant_initializer(initial))


class Ffnn(TfModel):
    def __init__(self, scope, hyper_parameters, extra, input_field, output_labels):
        super(Ffnn, self).__init__(scope)
        self.hyper_parameters = check.check_instance(hyper_parameters, HyperParameters)
        self.extra = extra
        self.input_field = check.check_instance(input_field, mlbase.Field)
        self.output_labels = check.check_instance(output_labels, mlbase.Labels)

        batch_size_dimension = None

        # Notation:
        #   _p      placeholder
        #   _c      constant

        # Base variable setup
        self.input_p = self.placeholder("input_p", [batch_size_dimension, len(self.input_field)])
        self.output_p = self.placeholder("output_p", [batch_size_dimension], tf.int32)
        self.learning_rate_p = self.placeholder("learning_rate_p", [1], tf.float32)
        self.clip_norm_p = self.placeholder("clip_norm_p", [1], tf.float32)
        self.dropout_keep_p = self.placeholder("dropout_keep_p", [1], tf.float32)

        if self.hyper_parameters.layers > 0:
            self.E = self.variable("E", [len(self.input_field), self.hyper_parameters.width])
            self.E_bias = self.variable("E_bias", [1, self.hyper_parameters.width], 0.)

            self.Y = self.variable("Y", [self.hyper_parameters.width, len(self.output_labels)])
            self.Y_bias = self.variable("Y_bias", [1, len(self.output_labels)], 0.)

            # The E layer is the first layer.
            if self.hyper_parameters.layers > 1:
                self.H = self.variable("H", [self.hyper_parameters.layers - 1, self.hyper_parameters.width, self.hyper_parameters.width])
                self.H_bias = self.variable("H_bias", [self.hyper_parameters.layers - 1, 1, self.hyper_parameters.width], 0.)

            # Computational graph encoding
            self.embedded_input = tf.tanh(tf.matmul(self.input_p, self.E) + self.E_bias)
            mlbase.assert_shape(self.embedded_input, [batch_size_dimension, self.hyper_parameters.width])
            hidden = self.embedded_input
            mlbase.assert_shape(hidden, [batch_size_dimension, self.hyper_parameters.width])

            for l in range(self.hyper_parameters.layers - 1):
                hidden = tf.tanh(tf.matmul(self.dropout(hidden), self.H[l]) + self.H_bias[l])
                mlbase.assert_shape(hidden, [batch_size_dimension, self.hyper_parameters.width])

            mlbase.assert_shape(hidden, [batch_size_dimension, self.hyper_parameters.width])
        else:
            self.Y = self.variable("Y", [len(self.input_field), len(self.output_labels)])
            self.Y_bias = self.variable("Y_bias", [1, len(self.output_labels)], 0.)

            # Computational graph encoding
            hidden = self.input_p
            mlbase.assert_shape(hidden, [batch_size_dimension, len(self.input_field)])

        self.output_logit = tf.matmul(self.dropout(hidden), self.Y) + self.Y_bias
        mlbase.assert_shape(self.output_logit, [batch_size_dimension, len(self.output_labels)])
        self.output_distributions = tf.nn.softmax(self.output_logit)
        mlbase.assert_shape(self.output_distributions, [batch_size_dimension, len(self.output_labels)])
        #self.cost = tf.reduce_sum(tf.nn.nce_loss(
        #    weights=tf.transpose(self.Y),
        #    biases=self.Y_bias,
        #    labels=self.output_p,
        #    inputs=hidden,
        #    num_sampled=1,
        #    num_classes=len(self.output_labels)))
        loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits
        self.cost = tf.reduce_sum(loss_fn(labels=tf.stop_gradient(self.output_p), logits=self.output_logit))
        #self.updates = tf.train.AdamOptimizer().minimize(self.cost)

        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate_p[0])
        gradients = optimizer.compute_gradients(self.cost)
        gradients_clipped = [(tf.clip_by_norm(g, self.clip_norm_p[0]), var) for g, var in gradients if g is not None]
        self.updates = optimizer.apply_gradients(gradients_clipped)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def dropout(self, tensor):
        return tf.nn.dropout(tensor, self.dropout_keep_p[0])

    def train(self, xys_stream, training_parameters):
        check.check_instance(training_parameters, mlbase.TrainingParameters)
        slot_length = len(str(training_parameters.epochs())) - 1
        epoch_template = "[%s] Epoch training {:%dd}: (loss, perplexity): {:.6f}, {:.6f}" % (self.scope, slot_length)
        final_loss = None
        epochs_tenth = max(1, int(training_parameters.epochs() / 10))
        losses = training_parameters.losses()
        finished = False
        epoch = -1

        while not finished:
            epoch += 1
            epoch_loss = 0
            # Start at a different offset for every epoch to help avoid overfitting.
            offset = random.randint(0, training_parameters.batch() - 1)
            batch = []
            first = True
            batch_set = False
            count = 0

            for xy in xys_stream():
                batch += [xy]

                if first and len(batch) == offset:
                    first = False
                    batch_set = True
                elif len(batch) == training_parameters.batch():
                    batch_set = True

                if batch_set:
                    count += len(batch)
                    xs = [self.input_field.vector_encode(xy.x, True) for xy in batch]
                    ys = [self.output_labels.encode(xy.y, True) for xy in batch]
                    feed = {
                        self.input_p: xs,
                        self.output_p: ys,
                        self.learning_rate_p: [training_parameters.learning_rate()],
                        self.clip_norm_p: [training_parameters.clip_norm()],
                        self.dropout_keep_p: [1.0 - training_parameters.dropout_rate()],
                    }
                    _, training_loss = self.session.run([self.updates, self.cost], feed_dict=feed)
                    epoch_loss += training_loss
                    batch_set = False
                    batch = []

            if len(batch) > 0:
                count += len(batch)
                xs = [self.input_field.vector_encode(xy.x, True) for xy in batch]
                ys = [self.output_labels.encode(xy.y, True) for xy in batch]
                feed = {
                    self.input_p: xs,
                    self.output_p: ys,
                    self.learning_rate_p: [training_parameters.learning_rate()],
                    self.clip_norm_p: [training_parameters.clip_norm()],
                    self.dropout_keep_p: [1.0 - training_parameters.dropout_rate()],
                }
                _, training_loss = self.session.run([self.updates, self.cost], feed_dict=feed)
                epoch_loss += training_loss

            epoch_loss /= count
            epoch_perplexity = math.exp(epoch_loss)
            losses.append(epoch_loss)
            finished, reason = training_parameters.finished(epoch, losses)

            if not finished and epoch % epochs_tenth == 0:
                logging.debug(epoch_template.format(epoch, epoch_loss, epoch_perplexity))

        logging.debug(epoch_template.format(epoch, epoch_loss, epoch_perplexity))
        logging.debug("Training on %d instances finished due to %s (%s)." % (count, reason, losses))
        return epoch_loss, -epoch_perplexity

    def converging_train(self, data_streams, model_dir, batch=32, arc_epochs=5, initial_decays=5, convergence_decays=2):
        assert initial_decays > convergence_decays, "%d <= %d" % (initial_decays, convergence_decays)
        train_stream, validation_stream, test_stream = data_streams
        best_score_train = self.test(train_stream)
        best_score_validation = self.test(validation_stream)
        score_test = self.test(test_stream)
        logging.debug("Baseline train/validation/test scores (random initialized weights): %.4f / %.4f / %.4f" % (best_score_train, best_score_validation, score_test))
        training_parameters = mlbase.TrainingParameters() \
            .batch(batch) \
            .epochs(arc_epochs) \
            .convergence(False) \
            .debug(True)
        previous_loss = None
        arc = -1
        version = 0
        self.save_parameters(model_dir, version, True)
        initialized = False
        converged = False
        decays = 0

        while not converged:
            arc += 1
            logging.debug("Train arc %d: %s" % (arc, training_parameters))
            loss, score_train, score_validation = self._train_loop(train_stream, validation_stream, training_parameters)
            loss_change = self._change(previous_loss, loss, lambda prev, curr: prev > curr)
            train_change = self._change(best_score_train, score_train, lambda prev, curr: prev < curr)
            validation_change = self._change(best_score_validation, score_validation, lambda prev, curr: prev < curr)
            logging.debug("Train arc %d: (loss, tr, va) (%s %.4f, %s %.4f, %s %.4f)" % (arc, loss_change, loss, train_change, score_train, validation_change, score_validation))
            both_improved = score_train > best_score_train and score_validation > best_score_validation

            if score_train > best_score_train or score_validation > best_score_validation:
                previous_loss = loss
                version += 1
                initialized = True
                self.save_parameters(model_dir, version, True)

                # At least one improved.
                if score_train > best_score_train:
                    best_score_train = score_train

                if score_validation > best_score_validation:
                    best_score_validation = score_validation
                else:
                    # The validation score didn't improve.  Lets see where the test score is at.
                    score_test = self.test(test_stream)
                    logging.debug("Test score: %.4f" % score_test)
            else:
                # Neither improved.
                # Load the best known version to continue training off of.
                self.load_parameters(model_dir)

            if not both_improved:
                if decays >= convergence_decays if initialized else decays >= initial_decays:
                    converged = True
                    logging.debug("Converged" + ("" if initialized else " without initialization!"))
                else:
                    decays += 1
                    logging.debug("Decaying.. %d", decays)
                    training_parameters = training_parameters.decay(initial=initialized)
            else:
                logging.debug("Reset decay")
                decays = 0

        # Load which ever version was marked as the latest as the final trained self.
        self.load_parameters(model_dir)
        logging.debug("Calculating final scores.")
        score_train = self.test(train_stream, False)
        score_validation = self.test(validation_stream, False)
        score_test = self.test(test_stream, True)
        logging.debug("(tr, va, te): (%.4f, %.4f, %.4f)" % (score_train, score_validation, score_test))

    def _change(self, previous, current, better_fn):
        if previous is None:
            return "-"
        elif better_fn(previous, current):
            return "▲"
        else:
            return "▼"

    def _train_loop(self, train_xys, validation_xys, training_parameters):
        loss, score_train = self.train(train_xys, training_parameters)
        score_validation = self.test(validation_xys)
        return loss, score_train, score_validation

    def evaluate(self, batch, handle_unknown=False):
        xs = [self.input_field.vector_encode(xy.x, handle_unknown) for xy in batch]
        ys = [self.output_labels.encode(xy.y, True) for xy in batch]
        feed = {
            self.input_p: xs,
            self.output_p: ys,
            self.dropout_keep_p: np.array([1.0]),
        }

        distributions, loss = self.session.run([self.output_distributions, self.cost], feed_dict=feed)

        if isinstance(batch, list):
            return [mlbase.Result(self.output_labels, distribution) for distribution in distributions], loss
        else:
            return mlbase.Result(self.output_labels, distributions[0]), loss

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


class CustomOutput(Model):
    def __init__(self, scope, output_labels, output_distribution):
        super(CustomOutput, self).__init__(scope)
        self.output_labels = check.check_instance(output_labels, mlbase.Labels)
        self.output_distribution = check.check_pdist(output_distribution)
        assert len(self.output_labels) == len(self.output_distribution), "%d != %d" % (len(self.output_labels), len(self.output_distribution))

    def evaluate(self, batch, handle_unknown=False):
        if isinstance(batch, list):
            return [mlbase.Result(self.output_labels, self.output_distribution) for i in range(len(batch))], None
        else:
            return mlbase.Result(self.output_labels, self.output_distribution), None


class HyperParameters:
    def __init__(self, layers, width):
        self.layers = layers
        self.width = width

    def __repr__(self):
        return "HyperParameters{l=%d, w=%d}" % (self.layers, self.width)

