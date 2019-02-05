#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import pdb
import re

from ml import base as mlbase
from pytils import check


# Tokens
START = "<start>"
END = "<end>"
UNKNOWN = "<unknown>"


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


def vocabulary(words):
    return mlbase.Labels(words.union(set([START, END])), unknown=UNKNOWN)


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


def auto_sentence_bleu(expected, actual):
    check.check_gte(len(expected), 1)
    ngrams = min(min(len(expected), len(actual)), 4)
    weight = 1.0 / ngrams
    return sentence_bleu([expected], actual, weights=tuple([weight] * ngrams))

