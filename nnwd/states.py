
import json
import os
import pdb

from ml import base as mlbase
from ml import nlp
from nnwd import semantic
from nnwd import parameters
from nnwd import pickler
from nnwd import view


STATES_TRAIN = "hidden-states-xys.train"
STATES_TEST = "hidden-states-xys.test"
STATES_ACTIVATION = "activation-states-xys"


def set_hidden_states(states_dir, is_train, key, states):
    return pickler.dump(states, os.path.join(states_dir, (STATES_TRAIN if is_train else STATES_TEST) + "." + key))


def get_hidden_states(states_dir, key):
    train = pickler.load(os.path.join(states_dir, STATES_TRAIN + "." + key))
    test = pickler.load(os.path.join(states_dir, STATES_TEST + "." + key))
    return train, test


def stream_hidden_train(states_dir, key, converter=None):
    return pickler.load(os.path.join(states_dir, STATES_TRAIN + "." + key), converter=converter)


def stream_hidden_test(states_dir, key, converter=None):
    return pickler.load(os.path.join(states_dir, STATES_TEST + "." + key), converter=converter)


def get_hidden_points(states_dir, key):
    width = view.part_width(key)

    def _as_point(data):
        # data is a tuple: (word, (padded array), annotation)
        return data[1][:width]

    train = pickler.load(os.path.join(states_dir, STATES_TRAIN + "." + key), converter=_as_point)
    test = pickler.load(os.path.join(states_dir, STATES_TEST + "." + key), converter=_as_point)
    return train, test


def set_activation_states(states_dir, key, states):
    return pickler.dump(states, os.path.join(states_dir, STATES_ACTIVATION + "." + key))


def get_activation_states(states_dir, key):
    return pickler.load(os.path.join(states_dir, STATES_ACTIVATION + "." + key))


def stream_activations(states_dir, key, converter=None):
    return pickler.load(os.path.join(states_dir, STATES_ACTIVATION + "." + key), converter=converter)

