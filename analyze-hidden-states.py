
from argparse import ArgumentParser
import collections
from csv import writer as csv_writer
import glob
import math
import logging
import os
import pdb
import queue
import random
import statistics
import sys

from nnwd import data
from nnwd import sequential
from nnwd import states
from nnwd import parameters
from nnwd import pickler
from nnwd import rnn

from pytils import adjutant
from pytils.log import setup_logging, teardown, user_log


@teardown
def main(argv):
    ap = ArgumentParser(prog="analyze-hidden-states")
    ap.add_argument("-v", "--verbose", default=False, action="store_true", help="Turn on verbose logging.")
    #ap.add_argument("-d", "--dry-run", default=False, action="store_true")
    ap.add_argument("--train-data", default=False, action="store_true")
    ap.add_argument("data_dir")
    ap.add_argument("sequential_dir")
    ap.add_argument("states_dir")
    aargs = ap.parse_args(argv)
    setup_logging(".%s.log" % os.path.splitext(os.path.basename(__file__))[0], aargs.verbose, False, True, True)
    logging.debug(aargs)

    lstm = sequential.load_model(aargs.data_dir, aargs.sequential_dir)
    stats = {}

    for key in lstm.keys():
        train_points, test_points = states.get_hidden_states(aargs.states_dir, key)

        if aargs.train_data:
            stats[key] = calculate_stats(train_points)
        else:
            stats[key] = calculate_stats(test_points)

    writer = csv_writer(sys.stdout)
    writer.writerow(["key"] + sorted(stats[next(iter(lstm.keys()))].keys()))
    averages = {}
    count = 0

    for key, stats in sorted(stats.items()):
        count += 1
        writer.writerow([key] + [item[1] for item in sorted(stats.items())])

        for key, value in stats.items():
            if key not in averages:
                averages[key] = 0

            averages[key] += value

    writer.writerow(["global"] + [item[1] / count for item in sorted(averages.items())])
    return 0


def calculate_stats(hidden_states):
    #flattened = adjutant.flat_map([[float(v) for v in hs.point] for hs in hidden_states])
    points = [[float(v) for v in hs.point] for hs in hidden_states]
    flattened = adjutant.flat_map(points)
    global_average = sum(flattened) / len(flattened)
    count = 0
    total = 0

    for point in points:
        count += 1
        total += sum([(v - global_average)**2 for v in point]) / len(point)

    return {"stdev": statistics.stdev(flattened), "mse": total / count}


if __name__ == "__main__":
    ret = main(sys.argv[1:])
    sys.exit(ret)

