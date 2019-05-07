
import json
import os

from ml import base as mlbase
from ml import nlp
from nnwd import parameters
from nnwd import pickler
from nnwd import view


LEARNED_BUCKETS = "buckets-learned"
FIXED_BUCKETS = "buckets-fixed"


def set_buckets(reduction_dir, key, learned_buckets, fixed_buckets):
    learned_path = os.path.join(reduction_dir, os.path.join(LEARNED_BUCKETS, key))
    fixed_path = os.path.join(reduction_dir, os.path.join(FIXED_BUCKETS, key))
    pickler.dump([item for item in learned_buckets.items()], learned_path)
    pickler.dump([item for item in fixed_buckets.items()], fixed_path)


def get_buckets(reduction_dir, key):
    learned_path = os.path.join(reduction_dir, os.path.join(LEARNED_BUCKETS, key))
    fixed_path = os.path.join(reduction_dir, os.path.join(FIXED_BUCKETS, key))
    learned = {item[0]: item[1] for item in pickler.load(learned_path)}
    fixed = {item[0]: item[1] for item in pickler.load(fixed_path)}
    return learned, fixed


def get_learned_buckets(reduction_dir):
    return {key: {item[0]: item[1] for item in pickler.load(os.path.join(reduction_dir, os.path.join(LEARNED_BUCKETS, key)))} for key in view.keys()}


def get_points(hs_dir, key):
    width = view.part_width(key)

    def _as_point(data):
        # data is a tuple: ((padded array), annotation)
        return data[0][:width]

    train = pickler.load(os.path.join(hs_dir, STATES_TRAIN + "." + key), converter=_as_point)
    test = pickler.load(os.path.join(hs_dir, STATES_TEST + "." + key), converter=_as_point)
    return train, test


def reduce(bucket_mapping, point):
    reduced = []

    for bucket, dimensions in sorted(bucket_mapping.items()):
        value = sum([point[dimension] for dimension in dimensions]) / len(dimensions)
        reduced += [value]

    return reduced


def mean_squared_error(bucket_mapping, point):
    error = 0.0

    for bucket, dimensions in sorted(bucket_mapping.items()):
        value = sum([point[dimension] for dimension in dimensions]) / len(dimensions)
        # Actual hidden state value compared to the reduced value.
        #              v                  v
        error += sum([(point[dimension] - value)**2 for dimension in dimensions])

    return error / len(point)

