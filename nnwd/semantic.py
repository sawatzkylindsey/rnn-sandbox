
import json
import os

from ml import base as mlbase
from ml import nlp
from ml import model
from nnwd import data
from nnwd import parameters
from nnwd import pickler
from nnwd import view

from pytils import check


SEM = "sem"

EMBEDDING_PADDING = tuple([0] * max(0, parameters.HIDDEN_WIDTH - parameters.EMBEDDING_WIDTH))
HIDDEN_PADDING = tuple([0] * max(0, parameters.EMBEDDING_WIDTH - parameters.HIDDEN_WIDTH))


def model_for(data_dir, model_fn):
    part_labels = mlbase.Labels(set(view.INSTRUMENTS))
    layer_labels = mlbase.Labels(set(range(parameters.LAYERS)))
    hidden_vector = mlbase.VectorField(max(parameters.HIDDEN_WIDTH, parameters.EMBEDDING_WIDTH))
    predictor_input = mlbase.ConcatField([part_labels, layer_labels, hidden_vector])
    description = data.get_description(data_dir)

    if description.task == data.LM:
        predictor_output = data.get_words(data_dir)
    else:
        predictor_output = data.get_outputs(data_dir)

    return model_fn("sem", predictor_input, predictor_output)


def load_model(model, encoding_dir):
    model.load(os.path.join(encoding_dir, SEM))


def save_model(model, encoding_dir):
    model.save(os.path.join(encoding_dir, SEM))


def as_input(key, point):
    part, layer = view.decode_key(key)
    is_embedding = part == "embedding"
    return (part, layer, tuple(point) + (EMBEDDING_PADDING if is_embedding else HIDDEN_PADDING))

