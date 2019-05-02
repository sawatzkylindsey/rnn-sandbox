
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
    ap = ArgumentParser(prog="generate-hidden-states")
    ap.add_argument("--verbose", "-v", default=False, action="store_true", help="Turn on verbose logging.")
    #ap.add_argument("-d", "--dry-run", default=False, action="store_true")
    ap.add_argument("states_dir")
    ap.add_argument("reduction_dir")
    ap.add_argument("target", type=int)
    aargs = ap.parse_args(argv)
    setup_logging(".%s.log" % os.path.splitext(os.path.basename(__file__))[0], aargs.verbose, False, True, True)
    logging.debug(aargs)
    part_learned_mse = {}
    part_fixed_mse = {}

    for key in view.part_keys():
        learned_mse, fixed_mse = generate_buckets(aargs.states_dir, key, aargs.reduction_dir, aargs.target)
        part_learned_mse[key] = learned_mse
        part_fixed_mse[key] = fixed_mse

    with open(os.path.join(aargs.reduction_dir, "dr-analysis-%d.csv" % aargs.target), "w") as fh:
        writer = csv_writer(fh)
        writer.writerow(["technique", "key", "mse"])

        for key, error in sorted(part_learned_mse.items()):
            writer.writerow(["learned", key, "%f" % error])

        for key, error in sorted(part_fixed_mse.items()):
            writer.writerow(["fixed", key, "%f" % error])

    return 0


def generate_buckets(states_dir, key, reduction_dir, target):
    logging.debug("Calculating for '%s'." % key)
    train_points, test_points = states.get_points(states_dir, key)
    width = view.part_width(key)
    learned_buckets, fixed_buckets = calculate_buckets(train_points, width, target)
    reduction.set_buckets(reduction_dir, target, key, learned_buckets, fixed_buckets)
    learned_mse = 0.0
    fixed_mse = 0.0
    count = 0

    for point in test_points:
        count += 1
        reduced, error = reduction.reduce(learned_buckets, point, calculate_mse=True)
        learned_mse += error
        reduced, error = reduction.reduce(fixed_buckets, point, calculate_mse=True)
        fixed_mse += error

    learned_mse /= count
    fixed_mse /= count
    logging.debug("'%s' with mse (learned, fixed): %.6f, %.6f" % (key, learned_mse, fixed_mse))
    return learned_mse, fixed_mse


def calculate_buckets(points, width, target):
    learned_buckets = {i: [] for i in range(target)}
    fixed_buckets = {i: [] for i in range(target)}
    fixed_size = math.ceil(float(width) / target)

    X = np.array([p for p in points]).transpose()
    gm = GaussianMixture(target)
    dimension_grouping = gm.fit_predict(X)
    fixed_group = 0

    for dimension, learned_group in enumerate(dimension_grouping):
        # If its the start of the transition to building out a new grouping of dimensions, and
        # if its the case that the remaining dimensions can be reduced evenly into a bucket of size less than 1, then do so.
        if len(fixed_buckets[fixed_group]) == 0 and fixed_size > 1 and len(fixed_buckets[fixed_group]) + ((width - dimension) / (fixed_size - 1)) == target:
            fixed_size -= 1

        learned_buckets[learned_group] += [dimension]
        fixed_buckets[fixed_group] += [dimension]

        if len(fixed_buckets[fixed_group]) == fixed_size:
            fixed_group += 1

    logging.debug("Buckets:\n  %s\n  %s" % (adjutant.dict_as_str(learned_buckets), adjutant.dict_as_str(fixed_buckets)))
    return learned_buckets, fixed_buckets


if __name__ == "__main__":
    ret = main(sys.argv[1:])
    sys.exit(ret)

