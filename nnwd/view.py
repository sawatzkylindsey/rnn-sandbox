
import json
import os

from ml import base as mlbase
from ml import nlp
from nnwd import parameters
from nnwd import pickler


INSTRUMENTS = [
    "embedding",
    "remember_gates",
    "forget_gates",
    "output_gates",
    "input_hats",
    "remembers",
    "cell_previouses",
    "forgets",
    "cells",
    "cell_hats",
    "outputs",
]
MATRICES = {
    "remember_gate": "R",
    "forget_gate": "F",
    "output_gate": "O",
    "input_hat": "H",
    "softmax": "Y",
}
LSTM_PARTS = [
    "remember_gates",
    "forget_gates",
    "output_gates",
    "input_hats",
    "remembers",
    "cell_previouses",
    "forgets",
    "cells",
    "cell_hats",
    "outputs",
]
SINGULAR = {
    "remember_gates": "remember_gate",
    "forget_gates": "forget_gate",
    "output_gates": "output_gate",
    "input_hats": "input_hat",
    "remembers": "remember",
    "cell_previouses": "cell_previous",
    "forgets": "forget",
    "cells": "cell",
    "cell_hats": "cell_hat",
    "outputs": "output",
}


def keys():
    keys = [encode_key("embedding")]

    for part in LSTM_PARTS:
        for layer in range(parameters.LAYERS):
            keys += [encode_key(part, layer)]

    return keys


def part_layers():
    parts = [("embedding", 0)]

    for part in LSTM_PARTS:
        for layer in range(parameters.LAYERS):
            parts += [(part, layer)]

    return parts


def is_embedding(part, layer):
    return part == "embedding"


def part_width(key):
    if key == encode_key("embedding"):
        return parameters.EMBEDDING_WIDTH
    else:
        return parameters.HIDDEN_WIDTH


def encode_key(part, layer=None):
    return "%s-%d" % (part, 0 if layer is None else layer)


def decode_key(key):
    result = key.split("-")

    if len(result) == 2:
        return (result[0], int(result[1]))
    else:
        raise ValueError("invalid decode of '%s': %s" % (key, result))


