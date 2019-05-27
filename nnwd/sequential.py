
import json
import os

from ml import base as mlbase
from ml import nlp
from ml import model
from nnwd import data
from nnwd import pickler
from nnwd import rnn


HYPER_PARAMETERS = "hyper-parameters.json"
ABLATIONS = "ablations.json"
LSTM = "lstm"


def get_hyper_parameters(sequential_dir):
    with open(os.path.join(sequential_dir, HYPER_PARAMETERS), "r") as fh:
        key_values = json.load(fh)
        hyper_parameters = HyperParameters.__new__(HyperParameters)
        hyper_parameters.__dict__ = {key: value for key, value in key_values.items()}
        return hyper_parameters


def set_hyper_parameters(sequential_dir, hyper_parameters):
    os.makedirs(sequential_dir, exist_ok=True)

    with open(os.path.join(sequential_dir, HYPER_PARAMETERS), "w") as fh:
        key_values = {key: value for key, value in hyper_parameters.__dict__.items()}
        fh.write(json.dumps(key_values, sort_keys=True, indent=4))


def get_ablations(sequential_dir):
    with open(os.path.join(sequential_dir, ABLATIONS), "r") as fh:
        key_values = json.load(fh)
        ablations = Ablations.__new__(Ablations)
        ablations.__dict__ = {key: value for key, value in key_values.items()}
        return ablations


def set_ablations(sequential_dir, ablations):
    os.makedirs(sequential_dir, exist_ok=True)

    with open(os.path.join(sequential_dir, ABLATIONS), "w") as fh:
        key_values = {key: value for key, value in ablations.__dict__.items()}
        fh.write(json.dumps(key_values, sort_keys=True, indent=4))


def model_for(data_dir, sequential_dir=None, hyper_parameters=None, ablations=None, skeleton=False):
    if sequential_dir is None:
        assert hyper_parameters is not None and ablations is not None, "one of (sequential_dir) or (hyper_parameters, ablations) must be specified"
    else:
        assert hyper_parameters is None and ablations is None, "both (sequential_dir) and (hyper_parameters, ablations) cannot be specified"
        hyper_parameters = get_hyper_parameters(sequential_dir)
        ablations = get_ablations(sequential_dir)

    description = data.get_description(data_dir)
    words = data.get_words(data_dir)

    if description.task == data.LM:
        return rnn.LstmLm(hyper_parameters, ablations, words, skeleton)
    else:
        outputs = data.get_outputs(data_dir)
        return rnn.LstmSa(hyper_parameters, ablations, words, outputs, skeleton)


def load_model(data_dir, sequential_dir, skeleton):
    rnn = model_for(data_dir, sequential_dir, skeleton=skeleton)

    if not skeleton:
        load_parameters(rnn, sequential_dir)

    return rnn


def load_parameters(rnn, sequential_dir):
    rnn.load_parameters(os.path.join(sequential_dir, LSTM))


def save_model(rnn, sequential_dir, version):
    set_hyper_parameters(sequential_dir, rnn.hyper_parameters)
    set_ablations(sequential_dir, rnn.ablations)
    save_parameters(rnn, sequential_dir, version)


def save_parameters(rnn, sequential_dir, version):
    rnn.save_parameters(os.path.join(sequential_dir, LSTM), version, True)


class HyperParameters:
    def __init__(self, layers, width, embedding_width):
        self.layers = layers
        self.width = width
        self.embedding_width = embedding_width


# Based off: LONG SHORT-TERM MEMORY AS A DYNAMICALLY COMPUTED ELEMENT-WISE WEIGHTED SUM (Levy*, Lee*, FitzGerald, Zettlemoyer 2018)
class Ablations:
    def __init__(self, srnn=False, out=False):
        self.srnn = srnn
        self.out = out

