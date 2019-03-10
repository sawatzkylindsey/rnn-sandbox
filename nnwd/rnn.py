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

        self.time_dimension = None
        self.batch_dimension = None
        # Notation:
        #   _p      placeholder
        #   _c      constant

        self.unrolled_inputs_p = self.placeholder("unrolled_inputs_p", [self.time_dimension, self.batch_dimension], tf.int32)
        self.input_lengths_p = self.placeholder("input_lengths_p", [self.batch_dimension], tf.int32)
        self.unrolled_outputs_p = self.placeholder("unrolled_outputs_p", [self.time_dimension, self.batch_dimension], tf.int32)

        self.initial_state_p = self.placeholder("initial_state_p", [Rnn.SCAN_STATES, self.layers, self.batch_dimension, self.width])

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
        assert_shape(self.unrolled_embedded_inputs, [self.time_dimension, self.batch_dimension, self.embedding_width])
        tf.identity(tf.reshape(self.unrolled_embedded_inputs, [self.embedding_width]), name="embedding")

        self.unrolled_embedded_projected_inputs = tf.matmul(tf.reshape(self.unrolled_embedded_inputs, [-1, self.embedding_width]), self.EP)
        assert_shape(self.unrolled_embedded_projected_inputs, [self.combine_dimensions(), self.width])

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
                assert_shape(output_previous[l], [self.batch_dimension, self.width])
                assert_shape(cell_previous[l], [self.batch_dimension, self.width])
                assert_shape(x, [self.batch_dimension, self.width])
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
                assert_shape(cell, [self.batch_dimension, self.width])
                cell_stack.append(cell)
                cell_hat = tf.tanh(cell)
                cell_hat_stack.append(cell_hat)
                output = cell_hat * output_gate
                assert_shape(output, [self.batch_dimension, self.width])
                output_stack.append(output)
                x = output

            return tf.stack([output_stack, cell_stack, remember_gate_stack, forget_gate_stack, output_gate_stack, input_hat_stack, remember_stack, cell_previous_stack, forget_stack, cell_hat_stack])

        max_time, batch_size = tf.unstack(tf.shape(self.unrolled_inputs_p))
        scan_inputs = tf.reshape(self.unrolled_embedded_projected_inputs, [max_time, batch_size, self.width])
        assert_shape(scan_inputs, [self.time_dimension, self.batch_dimension, self.width])
        self.unrolled_states = tf.scan(step_lstm, scan_inputs, self.initial_state_p)
        assert_shape(self.unrolled_states, [self.time_dimension, Rnn.SCAN_STATES, self.layers, self.batch_dimension, self.width])

        # Grab the last timesteps' state layers out of the unrolled state layers ([x y z] in diagram).
        # Notice, each cell in the diagram represents 2 (+all the extra instrumentation) states (hidden, cell, ..instrumentation..).
        #
        # state layer L     z z  z
        # ..                y y  y
        # state layer 0     x x  x
        # time              0 .. T
        self.state = self.unrolled_states[-1]
        assert_shape(self.state, [Rnn.SCAN_STATES, self.layers, self.batch_dimension, self.width])
        output_states, cell_states, remember_gates, forget_gates, output_gates, input_hats, remembers, cell_previouses, forgets, cell_hats = tf.unstack(self.state)
        tf.identity(tf.reshape(output_states, [-1, self.width]), name="outputs")
        tf.identity(tf.reshape(cell_states, [-1, self.width]), name="cells")
        tf.identity(tf.reshape(remember_gates, [-1, self.width]), name="remember_gates")
        tf.identity(tf.reshape(forget_gates, [-1, self.width]), name="forget_gates")
        tf.identity(tf.reshape(output_gates, [-1, self.width]), name="output_gates")
        tf.identity(tf.reshape(input_hats, [-1, self.width]), name="input_hats")
        tf.identity(tf.reshape(remembers, [-1, self.width]), name="remembers")
        tf.identity(tf.reshape(cell_previouses, [-1, self.width]), name="cell_previouses")
        tf.identity(tf.reshape(forgets, [-1, self.width]), name="forgets")
        tf.identity(tf.reshape(cell_hats, [-1, self.width]), name="cell_hats")

        # Grab the last state layer across all timesteps from the unrolled state layers ([z z z] in diagram).
        # Notice, each cell in the diagram represents 2 (+all the extra instrumentation) states (hidden, cell, ..instrumentation..).
        #
        # state layer L     z z  z
        # ..                y y  y
        # state layer 0     x x  x
        # time              0 .. T
        self.final_state = self.unrolled_states[:, 0, -1]
        assert_shape(self.final_state, [self.time_dimension, self.batch_dimension, self.width])

        final_state_for_matmul = tf.reshape(self.final_state, [-1, self.width])
        assert_shape(final_state_for_matmul, [self.combine_dimensions(), self.width])

        self.output_logits = tf.matmul(final_state_for_matmul, self.Y) + self.Y_bias
        assert_shape(self.output_logits, [self.combine_dimensions(), len(self.word_labels)])

        unrolled_outputs = tf.reshape(self.output_logits, [max_time, batch_size, len(self.word_labels)])
        assert_shape(unrolled_outputs, [self.time_dimension, self.batch_dimension, len(self.word_labels)])

        self.output_distributions = tf.nn.softmax(unrolled_outputs)
        assert_shape(self.output_distributions, [self.time_dimension, self.batch_dimension, len(self.word_labels)])

        expected_outputs = tf.reshape(self.output_logits, [max_time, batch_size, len(self.word_labels)])
        self.mask = tf.sequence_mask(self.input_lengths_p, dtype=tf.float32)
        self.cost = tf.contrib.seq2seq.sequence_loss(logits=expected_outputs, targets=self.unrolled_outputs_p, weights=self.mask)
        self.updates = tf.train.AdamOptimizer().minimize(self.cost)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def placeholder(self, name, shape, dtype=tf.float32):
        return tf.placeholder(dtype, shape, name=name)

    def variable(self, name, shape, dtype=tf.float32, initial=None):
        with tf.variable_scope(self.scope):
            return tf.get_variable(name, shape=shape, dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer() if initial is None else tf.constant_initializer(initial))

    def initial_state(self, batch_length):
        return np.zeros([Rnn.SCAN_STATES, self.layers, batch_length, self.width], dtype="float32")

    def combine_dimensions(self):
        if self.time_dimension is None or self.batch_dimension is None:
            return None
        else:
            return self.time_dimension * self.batch_dimension

    def train(self, xy_sequences, training_parameters):
        check.check_instance(training_parameters, mlbase.TrainingParameters)
        shuffled_xys = xy_sequences.copy()
        slot_length = len(str(training_parameters.epochs())) - 1
        epoch_template = "Epoch training {:%dd}: {:f}" % slot_length
        epochs_tenth = max(1, int(training_parameters.epochs() / 10))
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
                offset += training_parameters.batch()
                data_x, data_y = mlbase.as_time_major(batch, True)
                input_labels = [[self.word_labels.encode(word if word is not None else mlbase.BLANK) for word in timespot] for timespot in data_x]
                input_lengths = [len(sequence.x) for sequence in batch]
                output_labels = [[self.word_labels.encode(word if word is not None else mlbase.BLANK) for word in timespot] for timespot in data_y]
                parameters = {
                    self.unrolled_inputs_p: np.array(input_labels),
                    self.input_lengths_p: np.array(input_lengths),
                    self.initial_state_p: self.initial_state(len(batch)),
                    self.unrolled_outputs_p: np.array(output_labels),
                }
                _, total_cost = self.session.run([self.updates, self.cost], feed_dict=parameters)
                epoch_loss += total_cost

            if epoch % epochs_tenth == 0 and training_parameters.debug():
                logging.debug(epoch_template.format(epoch, epoch_loss))

            losses.append(epoch_loss)
            finished, reason = training_parameters.finished(epoch + 1, losses)

        logging.debug("Training finished due to %s (%s)." % (reason, losses))
        logging.debug(epoch_template.format(epoch, epoch_loss))
        return epoch_loss

    def test(self, xy_sequences, debug=False):
        assert len(xy_sequences) > 0
        total = 0.0
        case_slot_length = len(str(len(xy_sequences)))
        case_template = "{{Case {:%dd}}}" % case_slot_length
        offset = 0

        while offset < len(xy_sequences):
            batch = xy_sequences[offset:offset + 32]
            offset += 32
            data_x, data_y = mlbase.as_time_major(batch, True)
            input_labels = [[self.word_labels.encode(word if word is not None else mlbase.BLANK) for word in timespot] for timespot in data_x]
            input_lengths = [len(sequence.x) for sequence in batch]
            parameters = {
                self.unrolled_inputs_p: np.array(input_labels),
                self.initial_state_p: self.initial_state(len(batch)),
            }
            time_distributions = self.session.run(self.output_distributions, feed_dict=parameters)
            predictions = [[] for case in range(len(batch))]
            log_probabilities = [0.0 for case in range(len(batch))]

            for timestep, distributions in enumerate(time_distributions):
                for case, distribution in enumerate(distributions):
                    if timestep < input_lengths[case]:
                        predictions[case] += [self.word_labels.vector_decode(distribution)]
                        probability = self.word_labels.vector_decode_probability(distribution, batch[case].y[timestep])
                        log_probabilities[case] += math.log2(probability)

            for case, log_probability in enumerate(log_probabilities):
                perplexity = 2**(-(log_probability / input_lengths[case]))
                total += perplexity

                if debug:
                    logging.debug("%s perplexity %.4f.\n   Sequence: %s\n   Predicted: %s" % \
                        (case_template.format(case), perplexity, " ".join(batch[case].x), " ".join(predictions[case])))

        return total / len(xy_sequences)

    def evaluate(self, x, handle_unknown=False, state=None, instrument_names=[]):
        parameters = {
            self.unrolled_inputs_p: np.array([np.array([self.word_labels.encode(x, handle_unknown)])]),
            self.initial_state_p: state if state is not None else self.initial_state(1)
        }
        instruments = [self.session.graph.get_tensor_by_name("%s:0" % name) for name in instrument_names]
        distributions, next_state, *instrument_values = self.session.run([self.output_distributions, self.state] + instruments, feed_dict=parameters)
        assert len(distributions) == 1
        assert len(distributions[0]) == 1
        distribution = distributions[0][0]
        result = Result(self.word_labels.vector_decode(distribution), self.word_labels.vector_decode_distribution(distribution), self.word_labels.encoding())
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

    def stepwise(self, name=None, handle_unknown=False):
        return Stepwise(self, name, handle_unknown)

    def embed(self, x):
        parameters = {
            self.unrolled_inputs_p: np.array([np.array([self.word_labels.encode(x, True)])]),
            self.initial_state_p: self.initial_state(1)
        }
        return self.session.run([self.session.graph.get_tensor_by_name("embedding:0")], feed_dict=parameters)[0].tolist()

    def load(self, model_dir, version=None):
        checkpoints = Checkpoints.load(model_dir)
        model_path = checkpoints.model_path(version)
        version_key = "latest" if version is None else checkpoints.version_key(version)
        logging.debug("Restoring model %s=%s." % (version_key, model_path))
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope))
        saver.restore(self.session, model_path)

    def save(self, model_dir, version, set_latest=False):
        if os.path.isfile(model_dir) or (model_dir.endswith("/") and os.path.isfile(os.path.dirname(model_dir))):
            raise ValueError("model_dir '%s' must not be a file." % model_dir)

        checkpoints = Checkpoints.load(model_dir)

        if checkpoints is None:
            checkpoints = Checkpoints(model_dir)

        os.makedirs(model_dir, exist_ok=True)
        logging.debug("Saving model at %s=%d (latest=%s)." % (checkpoints.version_key(version), checkpoints.next_step, set_latest))
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope))
        saver.save(self.session, checkpoints.model_path_prefix(), global_step=checkpoints.next_step)
        checkpoints.update_next(version, set_latest) \
            .save()

    def mark_latest(self, model_dir, version):
        checkpoints = Checkpoints.load(model_dir)
        checkpoints.update_latest(version) \
            .save()


class Checkpoints:
    def __init__(self, model_dir, versions={}, latest=None, step=-1):
        self.model_dir = check.check_instance(model_dir, str)
        self.save_path = self.get_save_path(self.model_dir)
        self.versions = check.check_instance(versions, dict)
        self.latest = latest
        self.step = check.check_instance(step, int)
        self.next_step = self.step + 1

    def model_path_prefix(self):
        return os.path.join(self.model_dir, "basename")

    def version_key(self, version):
        return "v%s" % str(version)

    def model_path(self, version=None):
        if version is None:
            key = self.latest
        else:
            key = self.version_key(version)

        step = self.versions[key]
        return self.model_path_prefix() + ("-%d" % step)

    def update_next(self, version, set_latest=False):
        key = self.version_key(version)
        self.versions[key] = self.next_step
        self.step = self.next_step
        self.next_step += 1

        if self.latest is None or set_latest:
            self.latest = key

        return self

    def update_latest(self, version):
        self.latest = self.version_key(version)
        return self

    def as_json(self):
        return {
            "versions": self.versions,
            "latest": self.latest,
            "step": self.step,
        }

    def save(self):
        os.makedirs(self.model_dir, exist_ok=True)

        with open(self.save_path, "w") as fh:
            json.dump(self.as_json(), fh)

    @classmethod
    def get_save_path(self, model_dir):
        return os.path.join(model_dir, "checkpoints.json")

    @classmethod
    def load(self, model_dir):
        save_path = self.get_save_path(model_dir)

        if not os.path.exists(save_path):
            return None

        with open(save_path, "r") as fh:
            data = json.load(fh)
            return Checkpoints(model_dir, data["versions"], data["latest"], data["step"])


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

