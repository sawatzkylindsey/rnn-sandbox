
from argparse import ArgumentParser
import collections
from csv import writer as csv_writer
import glob
import logging
import os
import pdb
import queue
import random
import sys

from nnwd import data
from nnwd import sequential
from nnwd import states
from nnwd import parameters
from nnwd import pickler
from nnwd import rnn
from nnwd import sequential

from pytils import adjutant
from pytils.log import setup_logging, teardown, user_log


@teardown
def main(argv):
    ap = ArgumentParser(prog="generate-hidden-states")
    ap.add_argument("-v", "--verbose", default=False, action="store_true", help="Turn on verbose logging.")
    ap.add_argument("data_dir")
    ap.add_argument("sequential_dir")
    ap.add_argument("kind", choices=["train", "validation", "test"])
    ap.add_argument("dimensions", nargs="+", type=int)
    aargs = ap.parse_args(argv)
    setup_logging(".%s.log" % os.path.splitext(os.path.basename(__file__))[0], aargs.verbose, False, True, True)
    logging.debug(aargs)

    lstm = sequential.load_model(aargs.data_dir, aargs.sequential_dir)
    averages = categorize_rates(lstm, data.stream_data(aargs.data_dir, aargs.kind), aargs.dimensions)
    rows = [("", "0", "1")]

    for stat, dimension_points in averages.items():
        for dimension, points in dimension_points.items():
            rows += [("%s-%s" % (stat, dimension), *points)]

    with open("counter-statistics.csv", "w") as fh:
        writer = csv_writer(fh)

        for row in rows:
            writer.writerow(row)

    return 0


def categorize_rates(lstm, xys, dimensions):
    total = 0
    non_monotonic = 0
    starts = {
        "global": {dimension: 0 for dimension in dimensions},
        "monotonic": {dimension: 0 for dimension in dimensions},
        "non-monotonic": {dimension: 0 for dimension in dimensions},
    }
    ends = {
        "global": {dimension: 0 for dimension in dimensions},
        "monotonic": {dimension: 0 for dimension in dimensions},
        "non-monotonic": {dimension: 0 for dimension in dimensions},
    }
    global_lowest1 = {dimension: None for dimension in dimensions}
    global_lowest2 = {dimension: (None, None) for dimension in dimensions}

    for j, xy in enumerate(xys):
        sequence = [item[0] for item in xy.x]
        total += 1
        stepwise_rnn = lstm.stepwise(handle_unknown=True)
        cells = []
        previous = None
        index = None

        for i, word_pos in enumerate(xy.x):
            result, instruments = stepwise_rnn.step(word_pos[0], ["cells"])
            state = instruments["cells"][0]
            activations = [state[dimension] for dimension in dimensions]
            cells += [activations]

            if previous is not None and any([current < (previous[k] * 0.75) for k, current in enumerate(activations)]):
                index = i

            previous = activations

        for k, dimension in enumerate(dimensions):
            ck = [c[k] for c in cells]

            if global_lowest1[dimension] is None or lower1(global_lowest1[dimension], ck):
                global_lowest1[dimension] = ck

            if global_lowest2[dimension][0] is None or global_lowest2[dimension][0] < (sum(ck) / len(ck)):
                global_lowest2[dimension] = (sum(ck) / len(ck), ck)

            starts["global"][dimension] += cells[0][k]
            ends["global"][dimension] += cells[-1][k]

            if index is None:
                starts["monotonic"][dimension] += cells[0][k]
                ends["monotonic"][dimension] += cells[-1][k]
            else:
                starts["non-monotonic"][dimension] += cells[0][k]
                ends["non-monotonic"][dimension] += cells[-1][k]
                logging.debug("non-monotonic @%d (%s): %s -> %s" % (index, sequence[index], " ".join(sequence), " ".join([str(c) for c in cells])))
                non_monotonic += 1

    user_log.info("Found %d of %d sentences to match non-monotonic criteria." % (non_monotonic, total))
    user_log.info("Global lowest (by #1): %s" % (adjutant.dict_as_str(global_lowest1)))
    user_log.info("Global lowest (by #2): %s" % (adjutant.dict_as_str(global_lowest2)))
    averages = {}

    for stat in starts.keys():
        averages[stat] = {dimension: (starts[stat][dimension] / total, ends[stat][dimension] / total) for dimension in dimensions}

    return averages


def lower1(a, b):
    count = 0

    for i, v in enumerate(a):
        if i >= len(b):
            break

        if b[i] < v:
            count += 1

    return count == len(a)


if __name__ == "__main__":
    ret = main(sys.argv[1:])
    sys.exit(ret)

