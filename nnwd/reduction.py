
import json
import os

from ml import base as mlbase
from ml import nlp
from nnwd import parameters
from nnwd import pickler


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
    return _get_buckets(reduction_dir, LEARNED_BUCKETS)


def get_fixed_buckets(reduction_dir):
    return _get_buckets(reduction_dir, FIXED_BUCKETS)


def _get_buckets(reduction_dir, kind):
    buckets = {}

    for key in os.listdir(os.path.join(reduction_dir, kind)):
        buckets[key] = {item[0]: item[1] for item in pickler.load(os.path.join(reduction_dir, os.path.join(kind, key)))}

    return buckets


def reduce(bucket_mapping, point):
    reduced = []

    for bucket, dimensions in sorted(bucket_mapping.items()):
        value = sum([point[dimension] for dimension in dimensions]) / len(dimensions)
        reduced += [value]

    return reduced


def mean_squared_error(bucket_mapping, point):
    error = 0.0

    for bucket, dimensions in sorted(bucket_mapping.items()):
        if len(dimensions) == 0:
            value = 0.0
        else:
            value = sum([point[dimension] for dimension in dimensions]) / len(dimensions)

        # Actual hidden state value compared to the reduced value.
        #              v                  v
        error += sum([(point[dimension] - value)**2 for dimension in dimensions])

    return error / len(point)

