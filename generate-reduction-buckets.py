
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
from nnwd import parameters
from nnwd import pickler
from nnwd import reduction
from nnwd import rnn
from nnwd import sequential
from nnwd import states

from pytils import adjutant
from pytils.log import setup_logging, teardown, user_log


@teardown
def main(argv):
    ap = ArgumentParser(prog="generate-reduction-buckets")
    ap.add_argument("-v", "--verbose", default=False, action="store_true", help="Turn on verbose logging.")
    ap.add_argument("--grouping", nargs="*", default=None)
    ap.add_argument("data_dir")
    ap.add_argument("sequential_dir")
    ap.add_argument("states_dir")
    ap.add_argument("buckets_dir")
    ap.add_argument("target", type=int)
    aargs = ap.parse_args(argv)
    setup_logging(".%s.log" % os.path.splitext(os.path.basename(__file__))[0], aargs.verbose, False, True, True)
    logging.debug(aargs)

    lstm = sequential.load_model(aargs.data_dir, aargs.sequential_dir, True)
    part_learned_mse = {}
    part_fixed_mse = {}

    if aargs.grouping is None:
        for key in lstm.keys():
            learned_mse, fixed_mse = generate_buckets(aargs.states_dir, key, lstm.part_width(key), aargs.buckets_dir, aargs.target)
            part_learned_mse[key] = learned_mse
            part_fixed_mse[key] = fixed_mse
    else:
        learned_mse, fixed_mse = generate_buckets_grouping(lstm, aargs.states_dir, aargs.grouping, aargs.buckets_dir, aargs.target)
        part_learned_mse = learned_mse
        part_fixed_mse = fixed_mse

    with open(os.path.join(aargs.buckets_dir, "analysis.csv"), "w") as fh:
        writer = csv_writer(fh)
        writer.writerow(["technique", "key", "mse"])
        total_learned = 0.0
        total_fixed = 0.0
        count_learned = 0
        count_fixed = 0

        for key, error in sorted(part_learned_mse.items()):
            total_learned += error
            count_learned += 1
            writer.writerow(["learned", key, "%f" % error])

        for key, error in sorted(part_fixed_mse.items()):
            total_fixed += error
            count_fixed += 1
            writer.writerow(["fixed", key, "%f" % error])

        user_log.info("Total scores (learned, fixed): %s, %s" % (total_learned / count_learned, total_fixed / count_fixed))

    return 0


def generate_buckets(states_dir, key, width, buckets_dir, target):
    logging.debug("Calculating for '%s'." % key)
    train_points, test_points = states.get_hidden_states(states_dir, key)
    learned_buckets, fixed_buckets = calculate_buckets(width, target, [hidden_state.point for hidden_state in train_points])
    reduction.set_buckets(buckets_dir, key, learned_buckets, fixed_buckets)
    learned_mse = 0.0
    fixed_mse = 0.0
    count = 0

    for hidden_state in test_points:
        count += 1
        learned_mse += reduction.mean_squared_error(learned_buckets, hidden_state.point)
        fixed_mse += reduction.mean_squared_error(fixed_buckets, hidden_state.point)

    learned_mse /= count
    fixed_mse /= count
    logging.debug("tested %d instances of '%s' resulting in mse (learned, fixed): %.6f, %.6f" % (count, key, learned_mse, fixed_mse))
    return learned_mse, fixed_mse


def generate_buckets_grouping(lstm, states_dir, grouping, buckets_dir, target):
    logging.debug("Calculating for '%s'." % grouping)
    widths = [lstm.part_width(g) for g in grouping]
    width = widths[0]

    if any([w != width for w in widths]):
        raise ValueError("Cannot group across keys with difference widths: %s -> %s" % (grouping, widths))

    train_points = {}

    for key in grouping:
        train_stream, _ = states.get_hidden_states(states_dir, key)
        train_points[key] = [hidden_state.point for hidden_state in train_stream]

    learned_buckets = calculate_learned_grouping(width, target, train_points)
    fixed_buckets = calculate_fixed(width, target)
    learned_mses = {}
    fixed_mses = {}

    for key in lstm.keys():
        if lstm.part_width(key) == width:
            reduction.set_buckets(buckets_dir, key, learned_buckets, fixed_buckets)
            learned_mse = 0.0
            fixed_mse = 0.0
            count = 0

            for hidden_state in states.stream_hidden_test(states_dir, key):
                count += 1
                learned_mse += reduction.mean_squared_error(learned_buckets, hidden_state.point)
                fixed_mse += reduction.mean_squared_error(fixed_buckets, hidden_state.point)

            learned_mse /= count
            fixed_mse /= count
            logging.debug("tested %d instances of '%s' resulting in mse (learned, fixed): %.6f, %.6f" % (count, key, learned_mse, fixed_mse))
            learned_mses[key] = learned_mse
            fixed_mses[key] = fixed_mse

    return learned_mses, fixed_mses


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
    X_transpose = X.transpose()
    logging.debug("learning gaussian mixture model from %d points (%d wide)." % X.shape)
    dimension_grouping = None
    # This is the default reg_covar from sklearn.
    reg_covar = 1e-6

    while dimension_grouping is None:
        try:
            dimension_grouping = _gaussian_mixture(target, reg_covar, X_transpose)
        except ValueError as e:
            if "Fitting the mixture model failed" in str(e):
                reg_covar *= 10
                logging.debug("Fitting failed.. updating reg_covar=%s" % reg_covar)
            else:
                raise e

    for dimension, group in enumerate(dimension_grouping):
        buckets[group] += [dimension]

    return buckets


def calculate_learned_grouping(width, target, keyed_points):
    top_rate = 1.0
    buckets = None

    #while buckets is None:
    try:
        points = select(keyed_points, top_rate)
        buckets = calculate_learned(width, target, points)
    except Exception as e:
        raise e

    return buckets


def _gaussian_mixture(target, reg_covar, X_transpose):
    gm = GaussianMixture(target, covariance_type="diag", reg_covar=reg_covar)
    return gm.fit_predict(X_transpose)


def select(keyed_points, rate):
    for points in keyed_points.values():
        for point in points:
            if rate == 1.0 or random.random() <= rate:
                yield point


if __name__ == "__main__":
    ret = main(sys.argv[1:])
    sys.exit(ret)

