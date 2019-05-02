
import json
import os

from ml import base as mlbase
from ml import nlp
from nnwd.domain import NeuralNetwork
from nnwd import parameters
from nnwd import pickler


def part_keys():
    keys = [encode_key("embedding")]

    for part in NeuralNetwork.LSTM_PARTS:
        for layer in range(NeuralNetwork.LAYERS):
            keys += [encode_key(part, layer)]

    return keys


def part_width(key):
    if key == encode_key("embedding"):
        return NeuralNetwork.EMBEDDING_WIDTH
    else:
        return NeuralNetwork.HIDDEN_WIDTH


def encode_key(part, layer=None):
    return "%s-%d" % (part, 0 if layer is None else layer)


def decode_key(key):
    result = key.split("-")

    if len(result) == 2:
        return (result[0], int(result[1]))
    else:
        raise ValueError("invalid decode of '%s': %s" % (key, result))


