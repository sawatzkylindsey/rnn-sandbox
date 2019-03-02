#!/usr/bin/python
# -*- coding: utf-8 -*-

import collections
import logging
import math
import numpy as np
import pdb

from pytils import check


# Tasks
MULTI_LABEL = "multi-label"
SINGLE_LABEL = "single-label"

# Tokens
BLANK = "<blank>"


def as_time_major(xys, y_is_sequence=True):
    check.check_iterable_of_instances(xys, Xy)
    maximum_length_x = max([len(xy.x) for xy in xys])
    data_x = [[] for i in range(maximum_length_x)]

    if y_is_sequence:
        maximum_length_y = max([len(xy.y) for xy in xys])
        data_y = [[] for i in range(maximum_length_y)]
    else:
        data_y = []

    for j, xy in enumerate(xys):
        for i in range(maximum_length_x):
            if i < len(xy.x):
                data_x[i] += [xy.x[i]]
            else:
                data_x[i] += [None]

        if y_is_sequence:
            for i in range(maximum_length_y):
                if i < len(xy.y):
                    data_y[i] += [xy.y[i]]
                else:
                    data_y[i] += [None]
        else:
            data_y += [xy.y]

    return data_x, data_y


class Xy:
    def __init__(self, x, y, name=None):
        self.x = x
        self.y = y
        self._name = name

    def __repr__(self):
        return "(x=%s, y=%s)" % (self.x, self.y)

    def name(self):
        if self._name is None:
            return str(self)
        else:
            return self._name


class Result:
    def __init__(self, prediction, distribution):
        self.prediction = prediction
        self.distribution = distribution

    def __repr__(self):
        return "(prediction=%s, distribution=%s)" % (self.prediction, sorted(self.distribution.items()))


class TrainingParameters:
    DEFAULT_BATCH = 4
    DEFAULT_EPOCHS = 1000
    DEFAULT_ABSOLUTE = 0.05
    # .005%
    DEFAULT_RELATIVE = 0.00005
    # 25%
    DEFAULT_DEGRADATION = 0.25
    DEFAULT_WINDOW = 10
    DEFAULT_DEBUG = False
    REASON_EPOCHS = "maximum epochs"
    REASON_ABSOLUTE = "absolute convergence"
    REASON_RELATIVE = "relative convergence"
    REASON_DEGRADING = "degradation"

    def __init__(self):
        self._batch = TrainingParameters.DEFAULT_BATCH
        self._epochs = TrainingParameters.DEFAULT_EPOCHS
        self._absolute = TrainingParameters.DEFAULT_ABSOLUTE
        self._relative = TrainingParameters.DEFAULT_RELATIVE
        self._convergence = True
        self._degradation = TrainingParameters.DEFAULT_DEGRADATION
        self._window = TrainingParameters.DEFAULT_WINDOW
        self._debug = TrainingParameters.DEFAULT_DEBUG

    def losses(self):
        class Window:
            def __init__(self, size):
                self.size = size
                self.queue = collections.deque([])

            def append(self, value):
                self.queue.append(value)

                if len(self.queue) > self.size:
                    self.queue.popleft()

            def __iter__(self):
                return iter(self.queue)

            def __len__(self):
                return len(self.queue)

            def __getitem__(self, index):
                return self.queue[index]

            def __repr__(self):
                return str(["%.4f" % l for l in self])

        return Window(self._window)

    def finished(self, epoch, losses):
        # Training is finished if:
        #   1. The current epoch exceeds the epochs threshold, or
        #   2. The losses are full, decreasing, and consistently lower than the absolute threshold
        #   3. The losses are full, decreasing, and consistently lower than the relative threshold
        if epoch > self._epochs:
            return True, TrainingParameters.REASON_EPOCHS

        if len(losses) == self._window:
            if self._convergence:
                sliding_deltas = [losses[i] - losses[i + 1] for i in range(len(losses) - 1)]
                maximum_loss = max(losses)

                if all([d > 0.0 for d in sliding_deltas]):
                    if all([loss <= self._absolute for loss in losses]):
                        return True, TrainingParameters.REASON_ABSOLUTE

                    relative_threshold = maximum_loss * self._relative

                    if all([d <= relative_threshold for d in sliding_deltas]):
                        return True, TrainingParameters.REASON_RELATIVE

            first_deltas = [losses[0] - losses[i] for i in range(1, len(losses))]
            degradation_threshold = losses[0] * self._degradation

            # If the deltas between the first loss and the rest are all negative and exceeding the threshold, then maybe we're degraded.
            if all([delta <= -degradation_threshold for delta in first_deltas]):
                # If it looks like the degradation is recovering, then its OK.
                # That is, if the last 1/4 of the deltas are monotonically getting smaller then it seems to be recovering.
                for i in range(int(len(losses) * 0.75), len(losses) - 1):
                    if losses[i] >= losses[i + 1]:
                        logging.debug("Training experienced %s non-recovery (%s >= %s)." % (TrainingParameters.REASON_DEGRADING, losses[i], losses[i + 1]))
                        return True, TrainingParameters.REASON_DEGRADING

                logging.debug("Training experienced %s recovery (%s)." % (TrainingParameters.REASON_DEGRADING, losses))

        return False, None

    def batch(self, value=None):
        if value is None:
            return self._batch

        self._batch = check.check_gte(value, 1)
        return self

    def epochs(self, value=None):
        if value is None:
            return self._epochs

        self._epochs = check.check_gte(value, 1)
        return self

    def absolute(self, value=None):
        if value is None:
            return self._absolute

        self._absolute = check.check_gte(value, 0.0)
        return self

    def relative(self, value=None):
        if value is None:
            return self._relative

        self._relative = check.check_probability(value)
        return self

    def convergence(self, value=None):
        if value is None:
            return self._convergence

        self._convergence = check.check_one_of(value, [True, False])
        return self

    def degradation(self, value=None):
        if value is None:
            return self._degradation

        self._degradation = check.check_probability(value)
        return self

    def window(self, value=None):
        if value is None:
            return self._window

        self._window = check.check_gte(value, 1)
        return self

    def debug(self, value=None):
        if value is None:
            return self._debug

        self._debug = check.check_one_of(value, [True, False])
        return self

    def __repr__(self):
        return "TrainingParameters{b=%d, e=%d, a=%.4f, r=%.4f, r=%d, w=%d, d=%s}" % \
            (self._batch, self._epochs, self._absolute, self._relative, self._degradation, self._window, self._debug)


class Field(object):
    def __init__(self):
        super(Field, self).__init__()

    def encode(self, value, handle_unknown=False):
        raise NotImplementedError()

    def vector_encode(self, value, handle_unknown=False):
        raise NotImplementedError()

    def decode(self, value):
        raise NotImplementedError()

    def vector_decode(self, array):
        raise NotImplementedError()


class Labels(Field):
    def __init__(self, values, unknown=None):
        super(Labels, self).__init__()
        check.check_set(values)
        self.unknown = unknown
        self._empty = None
        self._encoding = {}
        self._decoding = {}
        labels_prefix = []

        if unknown is not None:
            self._encoding[unknown] = 0
            self._decoding[0] = self.unknown
            labels_prefix = [unknown]

        i = len(self._encoding)
        labels = sorted([label for label in values])

        for value in labels:
            self._encoding[check.check_not_none(value)] = i
            self._decoding[i] = value
            i += 1

        # Include unknown in the correct position if it's being represented in the labels.
        self._labels = labels_prefix + labels

    def __repr__(self):
        return "Labels{%s}" % self._encoding

    def __len__(self):
        return len(self._encoding)

    def encoding(self):
        return {k: v for k, v in self._encoding.items()}

    def labels(self):
        return self._labels

    def encode(self, value, handle_unknown=False):
        try:
            return self._encoding[check.check_not_none(value)]
        except KeyError as e:
            if handle_unknown:
                return self._encoding[self.unknown]
            else:
                raise e

    def vector_encode(self, value, handle_unknown=False):
        encoding = [0] * len(self)

        if isinstance(value, dict):
            for key, probability in check.check_pdist(value).items():
                encoding[self.encode(key, handle_unknown)] = probability
        else:
            encoding[self.encode(value, handle_unknown)] = 1

        return np.array(encoding, dtype="float32")

    def vector_empty(self):
        if self._empty is None:
            self._empty = np.array([0] * len(self))

        return self._empty

    def decode(self, value):
        return self._decoding[check.check_not_none(value)]

    def vector_decode(self, array):
        assert len(array) == len(self), "%d != %d" % (len(array), len(self))
        check.check_pdist(array)

        # If the array is all zeros
        if not np.any(array):
            return None

        # The index of the maximum value from the vector_encoding.
        #                  vvvvvvvvvvvvvv
        return self.decode(array.argmax())

    def sampling_vector_decode(self, array):
        assert len(array) == len(self), "%d != %d" % (len(array), len(self))
        check.check_pdist(array)
        # Sample 1 thing                    v
        #  from [0..N]           vvvvvvvvv
        #   with probabilties                  vvvvvvv
        index = np.random.choice(len(self), 1, p=array)[0]
        return self.decode(index)

    def vector_decode_distribution(self, array):
        assert len(array) == len(self), "%d != %d" % (len(array), len(self))
        check.check_pdist(array)
        return {self.decode(i): probability for i, probability in enumerate(array)}

    def vector_decode_probability(self, array, value):
        assert len(array) == len(self), "%d != %d" % (len(array), len(self))
        check.check_pdist(array)
        return array[self.encode(value)]


class VectorField(Field):
    def __init__(self, width):
        super(VectorField, self).__init__()
        self._length = width

    def __repr__(self):
        return "VectorField{%s}" % (self._length)

    def __len__(self):
        return self._length

    def encode(self, value, handle_unknown=False):
        raise TypeError()

    def vector_encode(self, value, handle_unknown=False):
        if len(value) != len(self):
            raise ValueError("value '%s' doesn't match vector width '%d'" % (value, self._length))

        return value

    def decode(self, value):
        raise TypeError()

    def vector_decode(self, array):
        raise TypeError()


class IntegerField(Field):
    def __init__(self):
        super(IntegerField, self).__init__()

    def __repr__(self):
        return "IntegerField"

    def __len__(self):
        return 1

    def encode(self, value, handle_unknown=False):
        raise TypeError()

    def vector_encode(self, value, handle_unknown=False):
        if not isinstance(value, int):
            raise ValueError("value '%s' isn't an integer" % value)

        return [value]

    def decode(self, value):
        raise TypeError()

    def vector_decode(self, array):
        raise TypeError()


class MergeLabels(Labels):
    def __init__(self, labels):
        super(MergeLabels, self).__init__(labels)
        self.labels = check.check_instance(labels, Labels)

    def __repr__(self):
        return "MergeLabels{%s}" % (self.labels)

    def __len__(self):
        return len(self.labels)

    def encoding(self):
        raise TypeError()

    def labels(self):
        raise TypeError()

    def encode(self, value, handle_unknown=False):
        raise TypeError()

    def vector_encode(self, value, handle_unknown=False):
        if len(value) > 0:
            # Produce the 'bitwise or' of the merge elements.
            return vector_max([self.labels.vector_encode(v, handle_unknown) for v in value])
        else:
            return self.labels.vector_empty()

    def decode(self, value):
        raise TypeError()

    def vector_decode(self, array):
        raise TypeError()

    def sampling_vector_decode(self, array):
        raise TypeError()

    def vector_decode_distribution(self, array):
        raise TypeError()


class ConcatField(Field):
    def __init__(self, fields):
        super(ConcatField, self).__init__()
        self.fields = check.check_iterable_of_instances(fields, Field)
        self._length = sum([len(f) for f in self.fields])

    def __repr__(self):
        return "ConcatField{%s}" % ([str(f) for f in self.fields])

    def __len__(self):
        return self._length

    def encoding(self):
        raise TypeError()

    def labels(self):
        raise TypeError()

    def encode(self, value, handle_unknown=False):
        raise TypeError()

    def vector_encode(self, value, handle_unknown=False):
        check.check_length(value, len(self.fields))
        return np.concatenate([self.fields[i].vector_encode(v, handle_unknown) for i, v in enumerate(value)])

    def decode(self, value):
        raise TypeError()

    def vector_decode(self, array):
        raise TypeError()

    def sampling_vector_decode(self, array):
        raise TypeError()

    def vector_decode_distribution(self, array):
        raise TypeError()


def assert_shape(tensor, expected):
    assert tensor.shape.as_list() == expected, "actual %s != expected %s" % (tensor.shape, expected)


def vector_max(vectors):
    length = None

    for ook in vectors:
        if length is None:
            length = len(ook)
        else:
            assert len(ook) == length, "%d != %d" % (len(ook), length)

    def _max(bits):
        out = bits[0]

        for bit in bits:
            if bit > out:
                out = bit

        return out

    out = np.array([_max([ook[i] for ook in vectors]) for i in range(length)])
    assert len(out) == length, "%d != %d" % (len(out), length)
    return out


def softmax(distribution):
    total = 0.0
    output = {}

    for k, v in distribution.items():
        value = math.exp(v)
        output[k] = value
        total += value

    return {k: v / total for k, v in output.items()}


def regmax(distribution):
    total = 0.0
    output = {}

    for k, v in distribution.items():
        value = v
        output[k] = value
        total += value

    return {k: v / total for k, v in output.items()}

