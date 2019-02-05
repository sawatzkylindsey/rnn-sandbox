#!/usr/bin/python
# -*- coding: utf-8 -*-

import collections
import numpy as np
import pdb

from pytils import check


# Tasks
MULTI_LABEL = "multi-label"
SINGLE_LABEL = "single-label"

# Tokens
BLANK = "<blank>"


def as_time_major(xys):
    check.check_iterable_of_instances(xys, Xy)
    maximum_length_x = max([len(xy.x) for xy in xys])
    maximum_length_y = max([len(xy.y) for xy in xys])
    data_x = [[] for i in range(maximum_length_x)]
    data_y = [[] for i in range(maximum_length_y)]

    for j, xy in enumerate(xys):
        for i in range(maximum_length_x):
            if i < len(xy.x):
                data_x[i] += [xy.x[i]]
            else:
                data_x[i] += [None]

        for i in range(maximum_length_y):
            if i < len(xy.y):
                data_y[i] += [xy.y[i]]
            else:
                data_y[i] += [None]

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
    DEFAULT_LOSS = 0.05
    DEFAULT_LOSS_WINDOW = 5
    DEFAULT_DEBUG = False

    def __init__(self):
        self._batch = TrainingParameters.DEFAULT_BATCH
        self._epochs = TrainingParameters.DEFAULT_EPOCHS
        self._loss = TrainingParameters.DEFAULT_LOSS
        self._loss_window = TrainingParameters.DEFAULT_LOSS_WINDOW
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

        return Window(self._loss_window)

    def finished(self, epoch, losses):
        # Training is finished if:
        #   1. The current epoch exceeds the epochs threshold, or
        #   2. The losses are full and consistently lower than the loss threshold
        return epoch > self._epochs \
            or (len(losses) == self._loss_window and all([loss < self._loss for loss in losses]))

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

    def loss(self, value=None):
        if value is None:
            return self._loss

        self._loss = check.check_gte(value, 0.0)
        return self

    def loss_window(self, value=None):
        if value is None:
            return self._loss_window

        self._loss_window = check.check_gte(value, 1)
        return self

    def debug(self, value=None):
        if value is None:
            return self._debug

        self._debug = check.check_one_of(value, [True, False])
        return self

    def __repr__(self):
        return "TrainingParameters{b=%d, e=%d, l=%.4f, lw=%d, d=%s}" % (self._batch, self._epochs, self._loss, self._loss_window, self._debug)


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

        if unknown is not None:
            self._encoding[unknown] = 0
            self._decoding[0] = self.unknown

        i = len(self._encoding)

        for value in values:
            self._encoding[check.check_not_none(value)] = i
            self._decoding[i] = value
            i += 1

        self._labels = sorted([label for label in self._encoding.keys()])

    def with_labels(self, values):
        new_labels = Labels.__new__(Labels)
        new_labels.unknown = self.unknown
        new_labels._empty = self._empty
        new_labels._encoding = self._encoding
        new_labels._decoding = self._decoding
        i = len(new_labels._encoding)

        for value in values:
            new_labels._encoding[check.check_not_none(value)] = i
            new_labels._decoding[i] = value
            i += 1

        new_labels._labels = sorted([label for label in new_labels._encoding.keys()])
        return new_labels

    def __repr__(self):
        return "Labels{%s}" % self._encoding

    def __len__(self):
        return len(self._encoding)

    def encodings(self):
        return {k: self._encoding[k] for k in sorted(self._encoding.keys())}

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
            raise ValueError()

        return value

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

    def encodings(self):
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

    def encodings(self):
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

