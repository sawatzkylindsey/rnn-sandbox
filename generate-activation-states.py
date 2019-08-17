
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
    ap = ArgumentParser(prog="generate-activation-states")
    ap.add_argument("-v", "--verbose", default=False, action="store_true", help="Turn on verbose logging.")
    #ap.add_argument("-d", "--dry-run", default=False, action="store_true")
    ap.add_argument("data_dir")
    ap.add_argument("sequential_dir")
    ap.add_argument("activations_dir")
    ap.add_argument("kind", choices=["train", "validation", "test"])
    aargs = ap.parse_args(argv)
    setup_logging(".%s.log" % os.path.splitext(os.path.basename(__file__))[0], aargs.verbose, False, True, True)
    logging.debug(aargs)

    lstm = sequential.load_model(aargs.data_dir, aargs.sequential_dir)
    description = data.get_description(aargs.data_dir)
    elicit_activation_states(lstm, data.stream_data(aargs.data_dir, aargs.kind), aargs.activations_dir)

    return 0


def elicit_activation_states(lstm, xys, activations_dir):
    activation_states = {}

    for key in lstm.keys():
        start_queue(activation_states, activations_dir, key)

    total = 0
    instances = 0

    for j, xy in enumerate(xys):
        total += 1
        instances += len(xy.x)
        stepwise_rnn = lstm.stepwise(handle_unknown=True)
        sequence = tuple(xy.x) + (xy.y[-1],)

        for i, word_pos in enumerate(xy.x):
            result, instruments = stepwise_rnn.step(word_pos[0], rnn.LSTM_INSTRUMENTS)

            for part, layer in lstm.part_layers():
                activation_states[lstm.encode_key(part, layer)].put(states.ActivationState(sequence, i, tuple([float(v) for v in instruments[part][layer]])))

    # Mark the queue as finished.
    for value in activation_states.values():
        value.put(None)

    user_log.info("%d sentences, eliciting %d activation states (per part-layer)." % (total, instances))


def start_queue(activation_states, activations_dir, key):
    states_queue = queue.Queue()
    activation_states[key] = states_queue
    states.set_activation_states(activations_dir, key, states_queue)


if __name__ == "__main__":
    ret = main(sys.argv[1:])
    sys.exit(ret)

