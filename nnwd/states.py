
import json
import os
import pdb

from ml import base as mlbase
from ml import nlp
from nnwd.domain import NeuralNetwork
from nnwd import semantic
from nnwd import parameters
from nnwd import pickler
from nnwd import view


STATES_TRAIN = "hidden-states-xys.train"
STATES_TEST = "hidden-states-xys.test"

EMBEDDING_PADDING = tuple([0] * max(0, NeuralNetwork.HIDDEN_WIDTH - NeuralNetwork.EMBEDDING_WIDTH))
HIDDEN_PADDING = tuple([0] * max(0, NeuralNetwork.EMBEDDING_WIDTH - NeuralNetwork.HIDDEN_WIDTH))


def as_point(array, is_embedding=False):
    return tuple(array) + (EMBEDDING_PADDING if is_embedding else HIDDEN_PADDING)


def set_states(states_dir, is_train, key, states):
    return pickler.dump(states, os.path.join(states_dir, (STATES_TRAIN if is_train else STATES_TEST) + "." + key))


def get_states(states_dir, key):
    train = pickler.load(os.path.join(states_dir, STATES_TRAIN + "." + key))
    test = pickler.load(os.path.join(states_dir, STATES_TEST + "." + key))
    return train, test


def stream_train(states_dir, key):
    return pickler.load(os.path.join(states_dir, STATES_TRAIN + "." + key), converter=_xy(key))


def stream_test(states_dir, key):
    return pickler.load(os.path.join(states_dir, STATES_TEST + "." + key), converter=_xy(key))


def _xy(key):
    def _fn(pair):
        return mlbase.Xy(semantic.as_input(key, pair[0]), pair[1])

    return _fn


def get_points(states_dir, key):
    width = view.part_width(key)

    def _as_point(data):
        # data is a tuple: ((padded array), annotation)
        return data[0][:width]

    train = pickler.load(os.path.join(states_dir, STATES_TRAIN + "." + key), converter=_as_point)
    test = pickler.load(os.path.join(states_dir, STATES_TEST + "." + key), converter=_as_point)
    return train, test

