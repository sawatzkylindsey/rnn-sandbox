
import collections
import json
import os
import pdb

from ml import base as mlbase
from ml import nlp
from nnwd import semantic
from nnwd import parameters
from nnwd import pickler


STATES_TRAIN = "hidden-states-xys.train"
STATES_TEST = "hidden-states-xys.test"
STATES_ACTIVATION = "activation-states-xys"
HiddenState = collections.namedtuple("HiddenState", ["word", "point", "annotation"])
ActivationState = collections.namedtuple("ActivationState", ["sequence", "index", "point"])


def set_hidden_states(states_dir, is_train, key, states):
    return pickler.dump(states, os.path.join(states_dir, (STATES_TRAIN if is_train else STATES_TEST) + "." + key), converter=lambda hs: tuple((hs.word, hs.point, hs.annotation)))


def get_hidden_states(states_dir, key):
    train = pickler.load(os.path.join(states_dir, STATES_TRAIN + "." + key), converter=lambda item: HiddenState(*item))
    test = pickler.load(os.path.join(states_dir, STATES_TEST + "." + key), converter=lambda item: HiddenState(*item))
    return train, test


def stream_hidden_train(states_dir):
    for name in os.listdir(states_dir):
        if name.startswith(STATES_TRAIN):
            for item in pickler.load(os.path.join(states_dir, name), converter=lambda item: HiddenState(*item)):
                yield _key(name), item


def _key(name):
    assert _is_key(name), "'%s' is not a 'key'" % name

    if name.startswith(STATES_TRAIN):
        return name[len(STATES_TRAIN) + 1:]
    else:
        return name[len(STATES_TEST) + 1:]


def _is_key(name):
    return name.startswith(STATES_TRAIN) or name.startswith(STATES_TEST)


def stream_hidden_test(states_dir, key):
    return pickler.load(os.path.join(states_dir, STATES_TEST + "." + key), converter=lambda item: HiddenState(*item))


def set_activation_states(states_dir, key, states):
    return pickler.dump(states, os.path.join(states_dir, STATES_ACTIVATION + "." + key), converter=lambda _as: tuple((_as.sequence, _as.index, _as.point)))


def get_activation_states(states_dir, key):
    return pickler.load(os.path.join(states_dir, STATES_ACTIVATION + "." + key), converter=lambda item: ActivationState(*item))


def stream_activations(states_dir, key):
    return pickler.load(os.path.join(states_dir, STATES_ACTIVATION + "." + key), converter=lambda item: ActivationState(*item))

