
import json
import os

from ml import base as mlbase
from ml import nlp
from nnwd import parameters
from nnwd import pickler


LEARNED_BUCKETS = "buckets-learned"
FIXED_BUCKETS = "buckets-fixed"


def as_point(array, is_embedding=False):
    return tuple(array) + (EMBEDDING_PADDING if is_embedding else HIDDEN_PADDING)


def set_buckets(reduction_dir, reduction, key, learned_buckets, fixed_buckets):
    learned_path = os.path.join(reduction_dir, os.path.join(LEARNED_BUCKETS + "." + str(reduction), key))
    fixed_path = os.path.join(reduction_dir, os.path.join(FIXED_BUCKETS + "." + str(reduction), key))
    pickler.dump([item for item in learned_buckets.items()], learned_path)
    pickler.dump([item for item in fixed_buckets.items()], fixed_path)


def get_buckets(reduction_dir, reduction, key):
    learned_path = os.path.join(reduction_dir, os.path.join(LEARNED_BUCKETS + "." + str(reduction), key))
    fixed_path = os.path.join(reduction_dir, os.path.join(FIXED_BUCKETS + "." + str(reduction), key))
    learned = {item[0]: item[1] for item in pickler.load(learned_path)}
    fixed = {item[0]: item[1] for item in pickler.load(fixed_path)}
    return learned, fixed


def get_points(hs_dir, key):
    width = view.part_width(key)

    def _as_point(data):
        # data is a tuple: ((padded array), annotation)
        return data[0][:width]

    train = pickler.load(os.path.join(hs_dir, STATES_TRAIN + "." + key), converter=_as_point)
    test = pickler.load(os.path.join(hs_dir, STATES_TEST + "." + key), converter=_as_point)
    return train, test


def reduce(bucket_mapping, point, calculate_mse=False):
    reduced = []
    error = 0.0

    for bucket, dimensions in sorted(bucket_mapping.items()):
        value = sum([point[dimension] for dimension in dimensions]) / len(dimensions)
        reduced += [value]

        if calculate_mse:
            # Actual hidden state value compared to the reduced value.
            #              v                  v
            error += sum([(point[dimension] - value)**2 for dimension in dimensions])

    return reduced, (error / len(point)) if calculate_mse else None

