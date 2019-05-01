
import json
import os

from ml import base as mlbase
from ml import nlp
from nnwd import parameters
from nnwd import pickler


LM = "lm"
SA = "sa"


def get_description(data_dir):
    with open(os.path.join(data_dir, parameters.DESCRIPTION), "r") as fh:
        key_values = json.load(fh)
        description = Description.__new__(Description)
        description.__dict__ = {key: value for key, value in key_values.items()}
        return description


def set_description(data_dir):
    with open(os.path.join(data_dir, parameters.DESCRIPTION), "w") as fh:
        fh.write(json.dumps(description, sort_keys=True, indent=4))


def get_words(data_dir):
    words = set([word for word in pickler.load(os.path.join(data_dir, parameters.WORDS))])
    return mlbase.Labels(words.union(set([mlbase.BLANK])), unknown=nlp.UNKNOWN)


def set_words(data_dir, words):
    pickler.dump([word for word in words], os.path.join(data_dir, parameters.WORDS))


def get_outputs(data_dir):
    outputs = set([outputs for output in pickler.load(os.path.join(data_dir, parameters.OUTPUTS))])
    return mlbase.Labels(outputs)


def set_outputs(data_dir, outputs, sort_key=lambda item: item):
    pickler.dump(sorted(outputs, key=sort_key), os.path.join(data_dir, parameters.OUTPUTS))


def stream_train(data_dir):
    description = get_description(data_dir)

    if description.task == LM:
        converter = _xy_lm
    elif description.task == SA:
        converter = _xy_sa
    else:
        raise ValueError()

    return pickler.load(os.path.join(data_dir, parameters.XYS_TRAIN), converter=converter)


def stream_test(data_dir):
    description = get_description(data_dir)

    if description.task == LM:
        converter = _xy_lm
    elif description.task == SA:
        converter = _xy_sa
    else:
        raise ValueError()

    return pickler.load(os.path.join(data_dir, parameters.XYS_TEST), converter=converter)


def _xy_sa(data):
    # data is a tuple: ([(word1, pos1), .., (wordN, posN)], sentiment)
    return mlbase.Xy(data[0], data[1])


def _xy_lm(data):
    # data is a sequence: [(word1, pos1), .., (wordN, posN)]
    if len(data) > 0:
        return mlbase.Xy(data[:-1], data[1:])
    else:
        return None


class Description:
    def __init__(self, task):
        self.task = task

    def __repr__(self):
        return "Description{%s}" % (self.task)

