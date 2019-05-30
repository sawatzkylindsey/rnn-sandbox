#!/usr/bin/python
# -*- coding: utf-8 -*-

import collections
import heapq
import json
import logging
import math
import numpy as np
import os
import pdb

from pytils import check


partial_sort_off = "partial_sort_off" in os.environ

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
    def __init__(self, labels, array):
        self.labels = labels
        self.array = array
        self._prediction = None
        self._distribution = None
        self._ranked_items = None
        self._rank_cache = {}

    def prediction(self):
        if self._prediction is None:
            self._prediction = self.labels.vector_decode(self.array)

        return self._prediction

    def distribution(self):
        if self._distribution is None:
            self._distribution = self.labels.vector_decode_distribution(self.array)

        return self._distribution

    def rank_of(self, value, handle_unknown=False, k=None):
        target = self.labels.decode(self.labels.encode(value, handle_unknown))

        if partial_sort_off:
            if self._ranked_items is None:
                self._ranked_items = {item[0]: rank for rank, item in enumerate(sorted(self.distribution().items(), key=lambda item: item[1], reverse=True))}

            return self._ranked_items[target]

        # Use a partial sort to find the rank.
        distribution_items = [item for item in self.distribution().items()]

        # Partial sort mechanism 'nlargest'.
        if k is not None:
            if self._ranked_items is None or len(self._ranked_items) < k:
                largest = heapq.nlargest(k, distribution_items, key=lambda item: item[1])
                self._ranked_items = {item[0]: rank for rank, item in enumerate(sorted(largest, key=lambda item: item[1], reverse=True))}

            if target in self._ranked_items:
                return self._ranked_items[target]
            else:
                # This is a lie - the nlargest method says the correct rank of the top-k elements, after
                # which everything else is given the last rank.
                return len(self.labels) - 1

        # Partial sort mechanism 'insertion-sort'.
        if target in self._rank_cache:
            return self._rank_cache[target]

        insertion_sorted = []
        partial_index = 0
        partial_total = 0
        index = 0
        rank = None

        while rank is None:
            encoding, probability = distribution_items[index]
            partial_total += probability
            insertion_index = binary_search(insertion_sorted, probability, accessor=lambda item: item[1])
            insertion_sorted.insert(insertion_index, (encoding, probability))

            if len(insertion_sorted) == len(distribution_items):
                # The entire list of items have been insertion sorted.
                # Find the target.
                while rank is None:
                    if partial_index >= len(insertion_sorted):
                        logging.info("something wrong for value '%s' target '%s' (handle unknown %s)" % (value, target, handle_unknown))

                    if insertion_sorted[partial_index][0] == target:
                        rank = partial_index

                    partial_index += 1
            else:
                # A new item has been insertion sorted.
                # Move up the partial index as much as is possible.
                remaining = 1.0 - partial_total

                # We can move up the partial index as long as the sum of the unknown portion of the probability distribution is less than
                # the probability at the current partial index.
                # This is true because we know that none of the unknown probabilities would exceed it (in which case they would need to be
                # insertion sorted in a way that changes the item at the partial index's rank).
                while partial_index < len(insertion_sorted) and remaining < insertion_sorted[partial_index][1]:
                    if insertion_sorted[partial_index][0] == target:
                        rank = partial_index
                        break

                    partial_index += 1

            index += 1

        self._rank_cache[target] = rank
        return self._rank_cache[target]

    def __repr__(self):
        return "(.., prediction=%s)" % (self.prediction())


class TrainingParameters:
    BATCH_DEFAULT = 32
    BATCH_MINIMUM = 4
    EPOCHS_DEFAULT = 10
    EPOCHS_MAXIMUM = 1000
    DROPOUT_RATE_DEFAULT = 0.1
    DROPOUT_RATE_MAXIMUM = 0.9
    LEARNING_RATE_DEFAULT = 1.0
    CLIP_NORM_DEFAULT = 5.0
    CLIP_NORM_MAXIMUM = 100
    DEFAULT_ABSOLUTE = 0.05
    # .005%
    DEFAULT_RELATIVE = 0.00005
    # 25%
    DEFAULT_DEGRADATION = 0.25
    DEFAULT_WINDOW = 10
    DEFAULT_DEBUG = False
    DEFAULT_SCORE = False
    REASON_EPOCHS = "maximum epochs"
    REASON_ABSOLUTE = "absolute convergence"
    REASON_RELATIVE = "relative convergence"
    REASON_DEGRADING = "degradation"

    def __init__(self):
        self._batch = TrainingParameters.BATCH_DEFAULT
        self._epochs = TrainingParameters.EPOCHS_DEFAULT
        self._dropout_rate = TrainingParameters.DROPOUT_RATE_DEFAULT
        self._learning_rate = TrainingParameters.LEARNING_RATE_DEFAULT
        self._clip_norm = TrainingParameters.CLIP_NORM_DEFAULT
        self._absolute = TrainingParameters.DEFAULT_ABSOLUTE
        self._relative = TrainingParameters.DEFAULT_RELATIVE
        self._convergence = True
        self._degradation = TrainingParameters.DEFAULT_DEGRADATION
        self._window = TrainingParameters.DEFAULT_WINDOW
        self._debug = TrainingParameters.DEFAULT_DEBUG
        self._score = TrainingParameters.DEFAULT_SCORE
        self._decays = 0

    def decay(self, initial=False):
        new_tp= TrainingParameters.__new__(TrainingParameters)
        new_tp.__dict__ = {k: v for k, v in self.__dict__.items()}
        new_tp._decays += 1
        new_tp._learning_rate /= 1.15

        if not initial:
            pass
            #new_tp._epochs = min(self._epochs + 1, TrainingParameters.EPOCHS_MAXIMUM)
            #new_tp._clip_norm = min(self._clip_norm * 2, TrainingParameters.CLIP_NORM_MAXIMUM)
            #new_tp._batch = max(int(self._batch * 0.8), TrainingParameters.BATCH_MINIMUM)
            #new_tp._dropout_rate = min(self._dropout_rate * 1.2, TrainingParameters.DROPOUT_RATE_MAXIMUM)

        return new_tp

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
        if epoch + 1 >= self._epochs:
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

    def dropout_rate(self, value=None):
        if value is None:
            return self._dropout_rate

        self._dropout_rate = check.check_gte(value, 0)
        return self

    def learning_rate(self, value=None):
        if value is None:
            return self._learning_rate

        self._learning_rate = check.check_gte(value, 0)
        return self

    def clip_norm(self, value=None):
        if value is None:
            return self._clip_norm

        self._clip_norm = check.check_gt(value, 0)
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

    def score(self, value=None):
        if value is None:
            return self._score

        self._score = check.check_one_of(value, [True, False])
        return self

    def __repr__(self):
        if self._convergence:
            return "TrainingParameters{btch=%d, epch=%d, drop=%.4f, learn=%.4f, cnrm=%.4f, abs-c=%.4f, rel-c=%.4f, degr=%d, wndw=%d, debug=%s}" % \
                (self._batch, self._epochs, self._dropout_rate, self._learning_rate, self._clip_norm, self._absolute, self._relative, self._degradation, self._window, self._debug)
        else:
            return "TrainingParameters{btch=%d, epch=%d, drop=%.4f, learn=%.4f, cnrm=%.4f, degr=%d, wndw=%d, debug=%s}" % \
                (self._batch, self._epochs, self._dropout_rate, self._learning_rate, self._clip_norm, self._degradation, self._window, self._debug)


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

    def copy(self, source, target, set_latest=False):
        source_key = self.version_key(source)
        target_key = self.version_key(target)
        self.versions[target_key] = self.versions[source_key]

        if self.latest is None or set_latest:
            self.latest = target_key

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
            if unknown is None or value != unknown:
                self._encoding[check.check_not_none(value)] = i
                self._decoding[i] = value
                i += 1

        assert len(self._encoding) == len(self._decoding), "%d != %d" % (len(self._encoding), len(self._decoding))
        # Include unknown in the correct position if it's being represented in the labels.
        self._labels = labels_prefix + labels
        self._encoding_copy = {k: v for k, v in self._encoding.items()}

    def __repr__(self):
        return "Labels{%s}" % self._encoding

    def __len__(self):
        return len(self._encoding)

    def encoding(self):
        return self._encoding_copy

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

        return np.array(encoding, dtype="float32", copy=False)

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

    def vector_decode_probability(self, array, value, handle_unknown=False):
        assert len(array) == len(self), "%d != %d" % (len(array), len(self))
        check.check_pdist(array)
        return array[self.encode(value, handle_unknown)]


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


# Return the index of the target, or where it should be inserted, based of an input array sorted descending.
def binary_search(descending_array, target, accessor=lambda item: item):
    if len(descending_array) == 0:
        return 0

    lower = 0
    upper = len(descending_array) - 1
    found = upper + 1 if accessor(descending_array[upper]) > target else None

    if found is None and accessor(descending_array[lower]) < target:
        found = 0

    while found is None:
        current = int((upper + lower) / 2.0)
        observation = accessor(descending_array[current])

        if observation == target:
            found = current + 1
        elif observation < target:
            upper = current
        else:
            lower = current

        if lower + 1 == upper:
            found = upper

    return found

