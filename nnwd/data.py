
import json
import os

from ml import base as mlbase
from ml import nlp
from nnwd import pickler


XYS_TRAIN = "xys.train"
XYS_VALIDATION = "xys.validation"
XYS_TEST = "xys.test"
WORDS = "words"
OUTPUTS = "outputs"
DESCRIPTION = "description.json"
OUTPUT_DISTRIBUTION = "output-distribution"
POS_MAPPING = "pos-mapping"
POS_TAGS = "pos-tags"

LM = "lm"
SA = "sa"

MINIMUM_OCCURRENCE_COUNT = 2


def get_description(data_dir):
    with open(os.path.join(data_dir, DESCRIPTION), "r") as fh:
        key_values = json.load(fh)
        description = Description.__new__(Description)
        description.__dict__ = {key: value for key, value in key_values.items()}
        return description


def set_description(data_dir, description):
    os.makedirs(data_dir, exist_ok=True)

    with open(os.path.join(data_dir, DESCRIPTION), "w") as fh:
        key_values = {key: value for key, value in description.__dict__.items()}
        fh.write(json.dumps(key_values, sort_keys=True, indent=4))


def get_words(data_dir):
    words = set([word for word in pickler.load(os.path.join(data_dir, WORDS))])
    return mlbase.Labels(words.union(set([mlbase.BLANK])), unknown=nlp.UNKNOWN)


def set_words(data_dir, words):
    pickler.dump([word for word in words], os.path.join(data_dir, WORDS))


def get_outputs(data_dir):
    outputs = set([output for output in pickler.load(os.path.join(data_dir, OUTPUTS))])
    return mlbase.Labels(outputs)


def set_outputs(data_dir, outputs, sort_key=lambda item: item):
    pickler.dump(sorted(outputs, key=sort_key), os.path.join(data_dir, OUTPUTS))


def get_output_distribution(data_dir):
    return {item[0]: item[1] for item in pickler.load(os.path.join(data_dir, OUTPUT_DISTRIBUTION))}


def set_output_distribution(data_dir, distribution):
    pickler.dump([(key, value) for key, value in distribution.items()], os.path.join(data_dir, OUTPUT_DISTRIBUTION))


def stream_train(data_dir):
    return stream_data(data_dir, "train")


def stream_validation(data_dir):
    return stream_data(data_dir, "validation")


def stream_test(data_dir):
    return stream_data(data_dir, "test")


def stream_data(data_dir, kind):
    description = get_description(data_dir)

    if description.task == LM:
        converter = _xy_lm
    elif description.task == SA:
        converter = _xy_sa
    else:
        raise ValueError()

    target_path = os.path.join(data_dir, XYS_TRAIN if kind == "train" else (XYS_TEST if kind == "test" else XYS_VALIDATION))
    return pickler.load(target_path, converter=converter)


def _xy_sa(data):
    # data is a tuple: ([(word1, pos1), .., (wordN, posN)], sentiment)
    return mlbase.Xy(data[0], data[1])


def _xy_lm(data):
    # data is a sequence: [(word1, pos1), .., (wordN, posN)]
    if len(data) > 1:
        return mlbase.Xy(data[:-1], data[1:])
    else:
        return None


def set_train(data_dir, pairs):
    _set_data(data_dir, pairs, "train")


def set_validation(data_dir, pairs):
    _set_data(data_dir, pairs, "validation")


def set_test(data_dir, pairs):
    _set_data(data_dir, pairs, "test")


def _set_data(data_dir, pairs, kind):
    target_path = os.path.join(data_dir, XYS_TRAIN if kind == "train" else (XYS_TEST if kind == "test" else XYS_VALIDATION))
    pickler.dump(pairs, target_path)


def set_pos_mapping(data_dir, pos_mapping):
    pickler.dump([item for item in pos_mapping.items()], os.path.join(data_dir, POS_MAPPING))


def get_pos_mapping(data_dir):
    return {item[0]: item[1] for item in pickler.load(os.path.join(data_dir, POS_MAPPING))}


def set_pos(data_dir, pos_tags):
    pickler.dump([pos for pos in pos_tags], os.path.join(data_dir, POS_TAGS))


class Description:
    def __init__(self, task):
        self.task = task

    def __repr__(self):
        return "Description{%s}" % (self.task)

