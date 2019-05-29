
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
    ap.add_argument("-s", "--sample-rate", type=float, default=0.1, help="train then test sampling rates.")
    ap.add_argument("-d", "--dry-run", default=False, action="store_true")
    ap.add_argument("data_dir")
    ap.add_argument("sequential_dir")
    ap.add_argument("states_dir")
    ap.add_argument("kind", choices=["train", "validation", "test"])
    aargs = ap.parse_args(argv)
    setup_logging(".%s.log" % os.path.splitext(os.path.basename(__file__))[0], aargs.verbose, False, True, True)
    logging.debug(aargs)

    if aargs.dry_run:
        dry_run(data.stream_data(aargs.data_dir, aargs.kind), aargs.sample_rate, aargs.kind)
        return 0

    lstm = sequential.load_model(aargs.data_dir, aargs.sequential_dir)
    description = data.get_description(aargs.data_dir)

    if description.task == data.LM:
        annotation_fn = lambda y, i: y[i][0]
    else:
        annotation_fn = lambda y, i: y

    elicit_hidden_states(lstm, data.stream_data(aargs.data_dir, aargs.kind), annotation_fn, aargs.sample_rate, aargs.states_dir, aargs.kind)
    return 0


def elicit_hidden_states(lstm, xys, annotation_fn, sample_rate, states_dir, kind):
    hidden_states = {}

    for key in lstm.keys():
        start_queue(hidden_states, states_dir, kind, key)

    total = 0
    sampled = 0
    instances = 0

    for j, xy in enumerate(xys):
        total += 1

        if random.random() <= sample_rate:
            sampled += 1
            instances += len(xy.x)
            stepwise_rnn = lstm.stepwise(handle_unknown=True)

            for i, word_pos in enumerate(xy.x):
                # Set the annotation to that which the rnn has been trained against, not the actual learned annotation (which will be fixed).
                # For example, consider the two training examples: "the little prince" -> "was" and "the little prince" -> "is".
                # We need predictor samples for both "was" and "is", but if we use the actual rnn annotation this will fixate on just one of these.
                annotation = annotation_fn(xy.y, i)
                result, instruments = stepwise_rnn.step(word_pos[0], rnn.LSTM_INSTRUMENTS)

                for part, layer in lstm.part_layers():
                    hidden_states[lstm.encode_key(part, layer)].put(states.HiddenState(word_pos[0], tuple([float(v) for v in instruments[part][layer]]), annotation))

    # Mark the queue as finished.
    for value in hidden_states.values():
        value.put(None)

    user_log.info("%s %.4f: %d sentences sampled down to %d, eliciting %d hidden states (per part-layer)." % (kind, sample_rate, total, sampled, instances))


def dry_run(xys, sample_rate, kind):
    total = 0
    sampled = 0
    instances = 0

    for j, xy in enumerate(xys):
        total += 1

        if random.random() <= sample_rate:
            sampled += 1
            instances += len(xy.x)

    user_log.info("(dry run) %s %.4f: %d sentences sampled down to %d, eliciting %d hidden states (per part-layer)." % (kind, sample_rate, total, sampled, instances))


def start_queue(hidden_states, states_dir, kind, key):
    states_queue = queue.Queue()
    hidden_states[key] = states_queue
    states.set_hidden_states(states_dir, kind, key, states_queue)


if __name__ == "__main__":
    ret = main(sys.argv[1:])
    sys.exit(ret)

