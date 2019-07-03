
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
    findings = find_non_counters(lstm, data.stream_data(aargs.data_dir, aargs.kind), aargs.dimensions)
    return 0


def find_non_counters(lstm, xys, dimensions):
    total = 0
    findings = []

    for j, xy in enumerate(xys):
        sequence = [item[0] for item in xy.x]
        total += 1
        stepwise_rnn = lstm.stepwise(handle_unknown=True)
        changes = []
        previous = None

        for i, word_pos in enumerate(xy.x):
            result, instruments = stepwise_rnn.step(word_pos[0], ["cells"])
            state = instruments["cells"][0]
            activations = [state[dimension] for dimension in dimensions]
            changes += [activations]

            if previous is not None and any([current < (previous[k] * 0.75) for k, current in enumerate(activations)]):
                findings += [(sequence, i)]
                logging.debug("non-counter @%d: %s -> %s" % (i, " ".join(sequence), " ".join([str(c) for c in changes])))
                break

            previous = activations

    user_log.info("Found %d of %d sentences to match non-counter criteria." % (len(findings), total))


if __name__ == "__main__":
    ret = main(sys.argv[1:])
    sys.exit(ret)

