
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

from pytils.log import setup_logging, teardown, user_log


@teardown
def main(argv):
    ap = ArgumentParser(prog="analyze-hidden-states")
    ap.add_argument("-v", "--verbose", default=False, action="store_true", help="Turn on verbose logging.")
    #ap.add_argument("-d", "--dry-run", default=False, action="store_true")
    ap.add_argument("data_dir")
    ap.add_argument("sequential_dir")
    ap.add_argument("states_dir")
    aargs = ap.parse_args(argv)
    setup_logging(".%s.log" % os.path.splitext(os.path.basename(__file__))[0], aargs.verbose, False, True, True)
    logging.debug(aargs)

    lstm = sequential.load_model(aargs.data_dir, aargs.sequential_dir)
    stats_train = {}
    stats_test = {}

    for key in lstm.keys():
        train_points, test_points = states.get_hidden_states(aargs.states_dir, key)
        stats_train[key] = calculate_stats(train_points)
        stats_test[key] = calculate_stats(test_points)

    writer = csv_writer(sys.stdout)
    writer.writerow(["dataset", "key", "minimum", "maximum"])
    global_minimum = None
    global_maximum = None

    for key, stats in sorted(stats_train.items()):
        writer.writerow(["train", key, stats["minimum"], stats["maximum"]])

        if global_minimum is None or stats["minimum"] < global_minimum:
            global_minimum = stats["minimum"]

        if global_maximum is None or stats["maximum"] > global_maximum:
            global_maximum = stats["maximum"]

    for key, stats in sorted(stats_test.items()):
        writer.writerow(["test", key, stats["minimum"], stats["maximum"]])

        if global_minimum is None or stats["minimum"] < global_minimum:
            global_minimum = stats["minimum"]

        if global_maximum is None or stats["maximum"] > global_maximum:
            global_maximum = stats["maximum"]

    writer.writerow(["global", "global", global_minimum, global_maximum])

    return 0


def calculate_stats(hidden_states):
    minimum = None
    maximum = None

    for hidden_state in hidden_states:
        local_minimum = min(hidden_state.point)

        if minimum is None or local_minimum < minimum:
            minimum = local_minimum

        local_maximum = max(hidden_state.point)

        if maximum is None or local_maximum > maximum:
            maximum = local_maximum

    return {"minimum": minimum, "maximum": maximum}



if __name__ == "__main__":
    ret = main(sys.argv[1:])
    sys.exit(ret)

