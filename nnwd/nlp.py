#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import math
import numpy as np
import pdb
import re

from pytils import check


START = "<start>"
END = "<end>"
UNKNOWN = "<unknown>"


class Labels:
    def __init__(self, values, unknown=None):
        check.check_instance(values, set)
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

    def __repr__(self):
        return "Labels{%s}" % self._encoding

    def __len__(self):
        return len(self._encoding)

    def encodings(self):
        return {k: self._encoding[k] for k in sorted(self._encoding.keys())}

    def labels(self):
        return set([v for v in self._encoding.keys()])

    def encode(self, value, handle_unknown=False):
        try:
            return self._encoding[check.check_not_none(value)]
        except KeyError as e:
            if handle_unknown:
                return self._encoding[self.unknown]
            else:
                raise e

    def ook_encode(self, value, handle_unknown=False):
        encoding = [0] * len(self)

        if isinstance(value, dict):
            for key, probability in check.check_pdist(value).items():
                encoding[self.encode(key, handle_unknown)] = probability
        else:
            encoding[self.encode(value, handle_unknown)] = 1

        return np.array(encoding, dtype="float32")

    def ook_empty(self):
        if self._empty is None:
            self._empty = np.array([0] * len(self))

        return self._empty

    def decode(self, value):
        return self._decoding[check.check_not_none(value)]

    def ook_decode(self, array):
        assert len(array) == len(self), "%d != %d" % (len(array), len(self))
        check.check_pdist(array)

        # If the array is all zeros
        if not np.any(array):
            return None

        # The index of the maximum value from the ook_encoding.
        #                  vvvvvvvvvvvvvv
        return self.decode(array.argmax())

    def sampling_ook_decode(self, array):
        assert len(array) == len(self), "%d != %d" % (len(array), len(self))
        check.check_pdist(array)
        # Sample 1 thing                    v
        #  from [0..N]           vvvvvvvvv
        #   with probabilties                  vvvvvvv
        index = np.random.choice(len(self), 1, p=array)[0]
        return self.decode(index)

    def ook_decode_distribution(self, array):
        assert len(array) == len(self), "%d != %d" % (len(array), len(self))
        check.check_pdist(array)
        return {self.decode(i): probability for i, probability in enumerate(array)}


class SpecialLabels:
    def __init__(self, value_labels, merge_labels):
        self.value_labels = value_labels
        self.merge_labels = merge_labels

    def __repr__(self):
        return "SpecialLabels{%s, %s}" % (self.value_labels._encoding, self.merge_labels._encoding)

    def __len__(self):
        return len(self.value_labels) + len(self.merge_labels)

    def encodings(self):
        raise TypeError()

    def labels(self):
        raise TypeError()

    def encode(self, value, handle_unknown=False):
        raise TypeError()

    def ook_encode(self, value, handle_unknown=False):
        value_part = self.value_labels.ook_encode(value[0], handle_unknown)

        if len(value[1]) > 0:
            # Produce the 'bitwise or' of the merge elements.
            merge_part = ook_max([self.merge_labels.ook_encode(v, handle_unknown) for v in value[1]])
        else:
            merge_part = self.merge_labels.ook_empty()

        return np.concatenate([value_part, merge_part])

    def decode(self, value):
        raise TypeError()

    def ook_decode(self, array):
        raise TypeError()

    def sampling_ook_decode(self, array):
        raise TypeError()

    def ook_decode_distribution(self, array):
        raise TypeError()


def split_words(text):
    return re.findall(r"\w+", text.lower(), re.UNICODE)


def split_sentences(text):
    words = split_words(text)
    sentences = []
    sentence = []

    for word in words:
        if word == END:
            sentences += [sentence]
            sentence = []
        else:
            sentence += [word]

    if len(sentence) > 0:
        sentences += [sentence]

    return sentences


def corpus_vocabulary(corpus_lines):
    words = set()

    for line in corpus_lines:
        for word in split_words(line):
            words.add(word)

    return vocabulary(words)


def corpus_sequences(corpus_file):
    corpus_lines = None

    with open(corpus_file, "r") as fh:
        corpus_lines = fh.readlines()

    words = set()
    xy_sequences = []

    for line in corpus_lines:
        for sentence in split_sentences(line):
            sequence = []

            for i, word in enumerate(sentence):
                words.add(word)

                if i + 1 < len(sentence):
                    sequence.append((word, sentence[i + 1]))

            if len(sequence) > 0:
                xy_sequences.append(sequence)

    labels = Labels(words, unknown=UNKNOWN)
    logging.info("words (%d): %s" % (len(labels), labels))
    return labels, xy_sequences


def vocabulary(words):
    return Labels(words.union(set([START, END])), unknown=UNKNOWN)


def left_to_right_window(vector, length, pad_value):
    assert length > 0
    out = None

    if len(vector) >= length:
        out = vector[len(vector) - length:]
    else:
        remaining = length - len(vector)
        out = ([pad_value] * remaining) + vector

    assert len(out) == length, "%d != %d" % (len(out), length)
    return out


def ook_max(ooks):
    length = None

    for ook in ooks:
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

    out = np.array([_max([ook[i] for ook in ooks]) for i in range(length)])
    assert len(out) == length, "%d != %d" % (len(out), length)
    return out


