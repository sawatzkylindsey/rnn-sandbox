
import collections
import json
import os
import pdb
import random

from ml import base as mlbase
from ml import nlp
from nnwd import semantic
from nnwd import parameters
from nnwd import pickler


STATES_TRAIN = "hidden-states-xys.train"
STATES_VALIDATION = "hidden-states-xys.validation"
STATES_TEST = "hidden-states-xys.test"
STATES_ACTIVATION = "activation-states-xys"
HiddenState = collections.namedtuple("HiddenState", ["word", "point", "annotation"])
ActivationState = collections.namedtuple("ActivationState", ["sequence", "index", "point"])


def _folder(kind):
    return STATES_TRAIN if kind == "train" else \
        STATES_VALIDATION if kind == "validation" else STATES_TEST


def set_hidden_states(states_dir, kind, key, states):
    pickler.dump(states, os.path.join(states_dir, _folder(kind) + "." + key), converter=lambda hs: tuple((hs.word, hs.point, hs.annotation)))


def get_hidden_states(states_dir, key):
    train = pickler.load(os.path.join(states_dir, STATES_TRAIN + "." + key), converter=lambda item: HiddenState(*item))
    test = pickler.load(os.path.join(states_dir, STATES_TEST + "." + key), converter=lambda item: HiddenState(*item))
    return train, test


def random_stream_hidden_states(states_dir, kind, keys, sample_rate=1.0):
    streams = {}
    stream_names = []

    for name in os.listdir(states_dir):
        key = _key(name)

        if name.startswith(_folder(kind)) and (keys is None or key in keys):
            streams[name] = pickler.load(os.path.join(states_dir, name), converter=lambda item: HiddenState(*item))
            stream_names += [name]

    while len(streams) > 0:
        name = random.choice(stream_names)

        try:
            item = next(streams[name])

            if sample_rate == 1.0 or random.random() <= sample_rate:
                yield _key(name), item
        except StopIteration as e:
            del streams[name]
            stream_names.remove(name)


def _key(name):
    assert _is_key(name), "'%s' is not a 'key'" % name

    if name.startswith(STATES_TRAIN):
        return name[len(STATES_TRAIN) + 1:]
    elif name.startswith(STATES_VALIDATION):
        return name[len(STATES_VALIDATION) + 1:]
    else:
        return name[len(STATES_TEST) + 1:]


def _is_key(name):
    return name.startswith(STATES_TRAIN) or name.startswith(STATES_TEST) or name.startswith(STATES_VALIDATION)


def stream_hidden_train(states_dir, key):
    return stream_hidden_states(states_dir, "test", key)


def stream_hidden_validation(states_dir, key):
    return stream_hidden_states(states_dir, "test", key)


def stream_hidden_test(states_dir, key):
    return stream_hidden_states(states_dir, "test", key)


def stream_hidden_states(states_dir, kind, key):
    return pickler.load(os.path.join(states_dir, _folder(kind) + "." + key), converter=lambda item: HiddenState(*item))


def set_activation_states(states_dir, key, states):
    pickler.dump(states, os.path.join(states_dir, STATES_ACTIVATION + "." + key), converter=lambda _as: tuple((_as.sequence, _as.index, _as.point)))


def get_activation_states(states_dir, key):
    return pickler.load(os.path.join(states_dir, STATES_ACTIVATION + "." + key), converter=lambda item: ActivationState(*item))


def stream_activations(states_dir, key):
    return pickler.load(os.path.join(states_dir, STATES_ACTIVATION + "." + key), converter=lambda item: ActivationState(*item))

