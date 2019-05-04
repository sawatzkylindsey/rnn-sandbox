
import json
import os

from ml import base as mlbase
from ml import nlp
from ml import nn as ffnn
from nnwd import data
from nnwd.domain import NeuralNetwork
from nnwd import parameters
from nnwd import pickler
from nnwd import view


def model_for(data_dir, layers, width):
    part_labels = mlbase.Labels(set(NeuralNetwork.INSTRUMENTS))
    layer_labels = mlbase.Labels(set(range(NeuralNetwork.LAYERS)))
    hidden_vector = mlbase.VectorField(max(NeuralNetwork.HIDDEN_WIDTH, NeuralNetwork.EMBEDDING_WIDTH))
    predictor_input = mlbase.ConcatField([part_labels, layer_labels, hidden_vector])
    description = data.get_description(data_dir)

    if description.task == data.LM:
        predictor_output = data.get_words(data_dir)
    else:
        predictor_output = data.get_outputs(data_dir)

    hyper_parameters = ffnn.HyperParameters() \
        .layers(layers) \
        .width(width)
    return ffnn.Model("sem", hyper_parameters, predictor_input, predictor_output)


def as_input(key, point):
    part, layer = view.decode_key(key)
    return (part, layer, tuple(point))

