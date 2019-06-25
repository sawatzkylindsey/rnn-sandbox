
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
import threading

from ml import base as mlbase
from ml import model
from ml import scoring
from nnwd import data
from nnwd import geometry
from nnwd import parameters
from nnwd import pickler
from nnwd import query
from nnwd import reduction
from nnwd import rnn
from nnwd import semantic
from nnwd import sequential
from nnwd import states

from pytils import adjutant
from pytils.log import setup_logging, teardown, user_log


BATCH_SIZE = 100

@teardown
def main(argv):
    ap = ArgumentParser(prog="measure-sequence-changes")
    ap.add_argument("-v", "--verbose", default=False, action="store_true", help="Turn on verbose logging.")
    ap.add_argument("data_dir")
    ap.add_argument("kind", choices=["train", "validation", "test"])
    ap.add_argument("sequential_dir")
    ap.add_argument("keys", nargs="+")
    aargs = ap.parse_args(argv)
    setup_logging(".%s.log" % os.path.splitext(os.path.basename(__file__))[0], aargs.verbose, False, True, True)
    logging.debug(aargs)

    lstm = sequential.load_model(aargs.data_dir, aargs.sequential_dir)
    minimum, maximum, sequence_changes = measure(lstm, aargs.data_dir, aargs.kind, aargs.keys)

    for key in aargs.keys:
        distance, index, sequence = minimum[key]
        sequence_str, changes_str = stringify(sequence, sequence_changes[sequence][key])
        user_log.info("Global minimum for %s of %.4f @%d:\n  %s\n  %s" % (key, distance, index, sequence_str, changes_str))
        distance, index, sequence = maximum[key]
        sequence_str, changes_str = stringify(sequence, sequence_changes[sequence][key])
        user_log.info("Global maximum for %s of %.4f @%d:\n  %s\n  %s" % (key, distance, index, sequence_str, changes_str))

    return 0


def measure(lstm, data_dir, kind, keys):
    sequence_changes = {}
    global_minimum = {key: (None, None, None) for key in keys}
    global_maximum = {key: (None, None, None) for key in keys}

    for j, xy in enumerate(data.stream_data(data_dir, kind)):
        if j % 100 == 0:
            logging.debug("At the %d instance." % (j))

        sequence = tuple([item[0] for item in xy.x]) + (xy.y[-1][0],)
        stepwise_rnn = lstm.stepwise(handle_unknown=True)
        change_distances = {key: [] for key in keys}
        previous_states = {}
        minimum = {key: (None, None) for key in keys}
        maximum = {key: (None, None) for key in keys}

        for i, word_pos in enumerate(xy.x):
            result, instruments = stepwise_rnn.step(word_pos[0], rnn.LSTM_INSTRUMENTS)

            for part, layer in lstm.part_layers():
                key = lstm.encode_key(part, layer)

                if key in keys:
                    current_state = instruments[part][layer]

                    if key in previous_states:
                        distance = geometry.distance(previous_states[key], current_state)
                    else:
                        distance = geometry.hypotenuse(current_state)

                    change_distances[key] += [distance]
                    previous_states[key] = current_state

                    if minimum[key] == (None, None) or distance < minimum[key][0]:
                        minimum[key] = (distance, i)

                    if maximum[key] == (None, None) or distance > maximum[key][0]:
                        maximum[key] = (distance, i)

        for key in keys:
            if global_minimum[key] == (None, None, None) or minimum[key][0] < global_minimum[key][0]:
                global_minimum[key] = minimum[key] + (sequence,)
                # Only keeping track of the more notable sequence changes
                sequence_changes[sequence] = change_distances
                sequence_str, changes_str = stringify(sequence, sequence_changes[sequence][key])
                logging.debug("Noting minimum for %s of %.4f @%d:\n  %s\n  %s" % (key, minimum[key][0], minimum[key][1], sequence_str, changes_str))

            if global_maximum[key] == (None, None, None) or maximum[key][0] > global_maximum[key][0]:
                global_maximum[key] = maximum[key] + (sequence,)
                # Only keeping track of the more notable sequence changes
                sequence_changes[sequence] = change_distances
                sequence_str, changes_str = stringify(sequence, sequence_changes[sequence][key])
                logging.debug("Noting maximum for %s of %.4f @%d:\n  %s\n  %s" % (key, maximum[key][0], maximum[key][1], sequence_str, changes_str))

    return global_minimum, global_maximum, sequence_changes


def stringify(sequence, changes):
    debug_template = " ".join(["{:%d.%ds}" % (max(len(word), 3), max(len(word), 3)) for word in sequence])
    float_template = "{:.4f}"
    sequence_str = debug_template.format(*sequence)
    changes_str = debug_template.format(*[float_template.format(c) for c in changes] + [""])
    return sequence_str, changes_str


if __name__ == "__main__":
    ret = main(sys.argv[1:])
    sys.exit(ret)

