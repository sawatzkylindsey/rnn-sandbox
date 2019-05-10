
from argparse import ArgumentParser
import collections
from csv import writer as csv_writer
import glob
import logging
import math
import numpy as np
import os
import pdb
import queue
import random
from sklearn.mixture import GaussianMixture
import sys

from nnwd import data
from nnwd.domain import NeuralNetwork
from nnwd import parameters
from nnwd import pickler
from nnwd import reduction
from nnwd import rnn
from nnwd import states
from nnwd import view

from pytils import adjutant
from pytils.log import setup_logging, user_log


def main(argv):
    ap = ArgumentParser(prog="generate-reduction-buckets")
    ap.add_argument("--verbose", "-v", default=False, action="store_true", help="Turn on verbose logging.")
    ap.add_argument("states_dir")
    ap.add_argument("buckets_dir")
    ap.add_argument("target", type=int)
    aargs = ap.parse_args(argv)
    setup_logging(".%s.log" % os.path.splitext(os.path.basename(__file__))[0], aargs.verbose, False, True, True)
    logging.debug(aargs)
    part_learned_mse = {}
    part_fixed_mse = {}

    for key in view.keys():
        learned_mse, fixed_mse = generate_buckets(aargs.states_dir, key, aargs.buckets_dir, aargs.target)
        part_learned_mse[key] = learned_mse
        part_fixed_mse[key] = fixed_mse

    with open(os.path.join(aargs.buckets_dir, "analysis.csv"), "w") as fh:
        writer = csv_writer(fh)
        writer.writerow(["technique", "key", "mse"])

        for key, error in sorted(part_learned_mse.items()):
            writer.writerow(["learned", key, "%f" % error])

        for key, error in sorted(part_fixed_mse.items()):
            writer.writerow(["fixed", key, "%f" % error])

    return 0


def generate_buckets(states_dir, key, buckets_dir, target):
    logging.debug("Calculating for '%s'." % key)
    train_points, test_points = states.get_hidden_points(states_dir, key)
    width = view.part_width(key)
    learned_buckets, fixed_buckets = calculate_buckets(width, target, train_points)
    reduction.set_buckets(buckets_dir, key, learned_buckets, fixed_buckets)
    learned_mse = 0.0
    fixed_mse = 0.0
    count = 0

    for point in test_points:
        count += 1
        learned_mse += reduction.mean_squared_error(learned_buckets, point)
        fixed_mse += reduction.mean_squared_error(fixed_buckets, point)

    learned_mse /= count
    fixed_mse /= count
    logging.debug("tested %d instances of '%s' resulting in mse (learned, fixed): %.6f, %.6f" % (count, key, learned_mse, fixed_mse))
    return learned_mse, fixed_mse


def calculate_buckets(width, target, points):
    fixed_buckets = calculate_fixed(width, target)
    learned_buckets = calculate_learned(width, target, points)
    logging.debug("Buckets:\n  %s\n  %s" % (adjutant.dict_as_str(learned_buckets), adjutant.dict_as_str(fixed_buckets)))
    return learned_buckets, fixed_buckets


def calculate_fixed(width, target):
    buckets = {i: [] for i in range(target)}
    size = math.ceil(float(width) / target)
    group = 0

    for dimension in range(width):
        # If its the start of the transition to building out a new grouping of dimensions, and
        # if its the case that the remaining dimensions can be reduced evenly into a bucket of size less than 1, then do so.
        if len(buckets[group]) == 0 and size > 1 and group + ((width - dimension) / (size - 1)) == target:
            size -= 1

        buckets[group] += [dimension]

        if len(buckets[group]) == size:
            group += 1

    return buckets


def calculate_learned(width, target, points):
    buckets = {i: [] for i in range(target)}
    X = np.array([p for p in points])
    logging.debug("learning gaussian mixture model from %d points (%d wide)." % X.shape)
    gm = GaussianMixture(target)
    dimension_grouping = gm.fit_predict(X.transpose())

    for dimension, group in enumerate(dimension_grouping):
        buckets[group] += [dimension]

    return buckets


if __name__ == "__main__":
    ret = main(sys.argv[1:])
    sys.exit(ret)

