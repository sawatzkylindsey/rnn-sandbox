
import json
import os

from ml import base as mlbase
from ml import nlp
from ml import model
from nnwd import data
from nnwd import parameters
from nnwd import pickler
from nnwd import rnn

from pytils import check


HYPER_PARAMETERS = "hyper-parameters.json"
EXTRA = "extra"
SEM = "sem"


def get_hyper_parameters(semantic_dir):
    with open(os.path.join(semantic_dir, HYPER_PARAMETERS), "r") as fh:
        key_values = json.load(fh)
        hyper_parameters = model.HyperParameters.__new__(model.HyperParameters)
        hyper_parameters.__dict__ = {key: value for key, value in key_values.items()}
        return hyper_parameters


def set_hyper_parameters(semantic_dir, hyper_parameters):
    os.makedirs(semantic_dir, exist_ok=True)

    with open(os.path.join(semantic_dir, HYPER_PARAMETERS), "w") as fh:
        key_values = {key: value for key, value in hyper_parameters.__dict__.items()}
        fh.write(json.dumps(key_values, sort_keys=True, indent=4))


def get_extra(semantic_dir):
    with open(os.path.join(semantic_dir, EXTRA), "r") as fh:
        return json.load(fh)


def set_extra(semantic_dir, extra):
    os.makedirs(semantic_dir, exist_ok=True)

    with open(os.path.join(semantic_dir, EXTRA), "w") as fh:
        fh.write(json.dumps(extra, sort_keys=True, indent=4))


def model_for(lstm, semantic_dir=None, hyper_parameters=None, extra=None, model_fn=None):
    if semantic_dir is None:
        assert hyper_parameters is not None and extra is not None, "one of (semantic_dir) or (hyper_parameters, extra) must be specified"
    else:
        assert hyper_parameters is None and extra is None, "both (semantic_dir) and (hyper_parameters, extra) cannot be specified"
        hyper_parameters = get_hyper_parameters(semantic_dir)
        extra = get_extra(semantic_dir)

    part_labels = mlbase.Labels(set(rnn.LSTM_INSTRUMENTS))
    layer_labels = mlbase.Labels(set(range(lstm.hyper_parameters.layers)))
    hidden_vector = mlbase.VectorField(max(lstm.hyper_parameters.width, lstm.hyper_parameters.embedding_width))

    embedding_padding = tuple([0] * max(0, lstm.hyper_parameters.width - lstm.hyper_parameters.embedding_width))
    hidden_padding = tuple([0] * max(0, lstm.hyper_parameters.embedding_width - lstm.hyper_parameters.width))

    if extra["word_input"]:
        input_field = mlbase.ConcatField([lstm.word_labels, part_labels, layer_labels, hidden_vector])

        def as_input(key, hidden_state):
            part, layer = lstm.decode_key(key)
            return (hidden_state.word, part, layer, tuple(hidden_state.point) + (embedding_padding if lstm.is_embedding(key) else hidden_padding))
    else:
        input_field = mlbase.ConcatField([part_labels, layer_labels, hidden_vector])

        def as_input(key, hidden_state):
            part, layer = lstm.decode_key(key)
            return (part, layer, tuple(hidden_state.point) + (embedding_padding if lstm.is_embedding(key) else hidden_padding))

    return model_fn(hyper_parameters, extra, input_field, lstm.output_labels, "sem"), as_input


def load_model(lstm, semantic_dir, model_fn):
    sem, as_input = model_for(lstm, semantic_dir, model_fn=model_fn)
    load_parameters(sem, semantic_dir)
    return sem, as_input


def load_parameters(sem, semantic_dir):
    sem.load_parameters(os.path.join(semantic_dir, SEM))


def save_model(sem, semantic_dir):
    set_hyper_parameters(semantic_dir, sem.hyper_parameters)
    set_extra(semantic_dir, sem.extra)
    save_parameters(sem, semantic_dir)


def save_parameters(sem, semantic_dir):
    sem.save_parameters(os.path.join(semantic_dir, SEM))

