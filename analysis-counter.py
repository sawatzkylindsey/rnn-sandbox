
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
    ap.add_argument("--report", default=False, action="store_true")
    ap.add_argument("data_dir")
    ap.add_argument("sequential_dir")
    ap.add_argument("kind", choices=["train", "validation", "test"])
    ap.add_argument("dimensions", nargs="+", type=int)
    aargs = ap.parse_args(argv)
    setup_logging(".%s.log" % os.path.splitext(os.path.basename(__file__))[0], aargs.verbose, False, True, True)
    logging.debug(aargs)

    lstm = sequential.load_model(aargs.data_dir, aargs.sequential_dir)
    averages = categorize_rates(lstm, data.stream_data(aargs.data_dir, aargs.kind), aargs.dimensions, aargs.report)
    rows = [("", "0", "1")]

    for stat, dimension_points in averages.items():
        for dimension, points in dimension_points.items():
            rows += [("%s-%s" % (stat, dimension), *points)]

    with open("counter-statistics.csv", "w") as fh:
        writer = csv_writer(fh)

        for row in rows:
            writer.writerow(row)

    return 0


def categorize_rates(lstm, xys, dimensions, report):
    total = 0
    non_monotonic = 0
    non_monotonic_counts = {}
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
    global_lowest1 = {dimension: (None, None, None) for dimension in dimensions}
    global_lowest2 = {dimension: (None, None, None) for dimension in dimensions}
    global_lowest3 = {dimension: (None, None, None) for dimension in dimensions}
    largest_drop = {dimension: (None, None, None) for dimension in dimensions}
    minimum_growth = {dimension: (None, None, None) for dimension in dimensions}

    for j, xy in enumerate(xys):
        if j % 1000 == 0:
            logging.debug("At the %d-Kth instance." % (int(j / 1000)))

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

            if global_lowest1[dimension][0] is None or lower1(ck, global_lowest1[dimension][1]):
                global_lowest1[dimension] = ("moot", ck, sequence)

            if global_lowest2[dimension][0] is None or (sum(ck) / len(ck)) < global_lowest2[dimension][0]:
                global_lowest2[dimension] = (sum(ck) / len(ck), ck, sequence)

            if global_lowest3[dimension][0] is None or min(ck) < global_lowest3[dimension][0]:
                global_lowest3[dimension] = (min(ck), ck, sequence)

            for i in range(len(ck) - 1):
                if largest_drop[dimension][0] is None or (ck[i + 1] - ck[i]) < largest_drop[dimension][0]:
                    largest_drop[dimension] = (ck[i + 1] - ck[i], ck, sequence)

            if len(ck) > 1:
                if minimum_growth[dimension][0] is None or (ck[-1] - ck[0]) < minimum_growth[dimension][0]:
                    minimum_growth[dimension] = (ck[-1] - ck[0], ck, sequence)

            starts["global"][dimension] += cells[0][k]
            ends["global"][dimension] += cells[-1][k]

            if index is None:
                starts["monotonic"][dimension] += cells[0][k]
                ends["monotonic"][dimension] += cells[-1][k]
            else:
                starts["non-monotonic"][dimension] += cells[0][k]
                ends["non-monotonic"][dimension] += cells[-1][k]

        if index is not None:
            if report:
                logging.debug("non-monotonic @%d (%s): %s -> %s" % (index, sequence[index], " ".join(sequence), " ".join([str(c) for c in cells])))

            non_monotonic += 1

            if sequence[index] not in non_monotonic_counts:
                non_monotonic_counts[sequence[index]] = 0

            non_monotonic_counts[sequence[index]] += 1

    user_log.info("Found %d of %d sentences to match non-monotonic criteria." % (non_monotonic, total))
    user_log.info("Non-monotonic keyword frequencies: %s" % (adjutant.dict_as_str(non_monotonic_counts, sort_by_key=False, reverse=True)))

    for dimension in dimensions:
        user_log.info("Global lowest @%d (by progression): %s" % (dimension, global_lowest1[dimension]))
        user_log.info("Global lowest @%d (by average): %s" % (dimension, global_lowest2[dimension]))
        user_log.info("Global lowest @%d (by single minimum): %s" % (dimension, global_lowest3[dimension]))
        user_log.info("Global lowest @%d (by largest drop): %s" % (dimension, largest_drop[dimension]))
        user_log.info("Global lowest @%d (by minimum growth): %s" % (dimension, minimum_growth[dimension]))

    averages = {
        "global": {dimension: (starts["global"][dimension] / total, ends["global"][dimension] / total) for dimension in dimensions},
        "monotonic": {dimension: (starts["monotonic"][dimension] / (total - non_monotonic), ends["monotonic"][dimension] / (total - non_monotonic)) for dimension in dimensions},
        "non-monotonic": {dimension: (starts["non-monotonic"][dimension] / non_monotonic, ends["non-monotonic"][dimension] / non_monotonic) for dimension in dimensions},
    }
    logging.debug(adjutant.dict_as_str(averages))
    return averages


def lower1(a, b):
    count = 0

    for i, v in enumerate(b):
        if i >= len(a):
            break

        if a[i] < v:
            count += 1

    return count == len(b)


if __name__ == "__main__":
    ret = main(sys.argv[1:])
    sys.exit(ret)

