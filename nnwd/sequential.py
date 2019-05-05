
import json
import os

from ml import base as mlbase
from ml import nlp
from ml import model
from nnwd import data
from nnwd.domain import NeuralNetwork
from nnwd import parameters
from nnwd import pickler
from nnwd import rnn
from nnwd import view


LSTM = "lstm"


def model_for(data_dir):
    description = data.get_description(data_dir)
    words = data.get_words(data_dir)

    if description.task == data.LM:
        return rnn.RnnLm(NeuralNetwork.LAYERS, NeuralNetwork.HIDDEN_WIDTH, NeuralNetwork.EMBEDDING_WIDTH, words)
    else:
        outputs = data.get_outputs(data_dir)
        return rnn.RnnSa(NeuralNetwork.LAYERS, NeuralNetwork.HIDDEN_WIDTH, NeuralNetwork.EMBEDDING_WIDTH, words, outputs)


def load_model(rnn, sequential_dir):
    rnn.load(os.path.join(sequential_dir, LSTM))


def save_model(rnn, sequential_dir, version):
    rnn.save(os.path.join(sequential_dir, LSTM), version, True)


def as_input(key, point):
    part, layer = view.decode_key(key)
    return (part, layer, tuple(point))

