
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

from ml import base as mlbase
from ml import scoring
from nnwd import data
from nnwd.domain import NeuralNetwork
from nnwd import encoding
from nnwd import parameters
from nnwd import pickler
from nnwd import reduction
from nnwd import rnn
from nnwd import states
from nnwd import view

from pytils import adjutant
from pytils.log import setup_logging, user_log


SCORES = {
    "accuracy": scoring.accuracy,
    "rank_score_linear": scoring.rank_score_linear(),
    "rank_score_exponential": scoring.rank_score_exponential(),
    "top_k3": scoring.descrete_rank(top_k=2),
    "top_percent05": scoring.descrete_rank(top_percent=0.05),
    "top_percent10": scoring.descrete_rank(top_percent=0.1),
    "top_percent25": scoring.descrete_rank(top_percent=0.25),
}


def main(argv):
    ap = ArgumentParser(prog="generate-encodings")
    ap.add_argument("--verbose", "-v", default=False, action="store_true", help="Turn on verbose logging.")
    #ap.add_argument("-d", "--dry-run", default=False, action="store_true")
    ap.add_argument("-e", "--epochs", default=10, type=int)
    ap.add_argument("-l", "--layers", default=2, type=int)
    ap.add_argument("-w", "--width", default=100, type=int)
    ap.add_argument("data_dir")
    ap.add_argument("states_dir")
    ap.add_argument("encoding_dir")
    aargs = ap.parse_args(argv)
    setup_logging(".%s.log" % os.path.splitext(os.path.basename(__file__))[0], aargs.verbose, False, True, True)
    logging.debug(aargs)
    sem = encoding.model_for(aargs.data_dir, aargs.layers, aargs.width)
    key_scores = generate_model(sem, aargs.states_dir, aargs.encoding_dir, aargs.epochs)

    with open(os.path.join(aargs.encoding_dir, "analysis.csv"), "w") as fh:
        writer = csv_writer(fh)
        writer.writerow(["key", "score_fn", "result"])

        for key, scores in sorted(key_scores.items()):
            for name, score in sorted(scores.items()):
                writer.writerow([key, name, "%f" % score])

    return 0


def generate_model(sem, states_dir, encoding_dir, epochs):
    train_xys = []

    for key in view.part_keys():
        for xy in states.stream_train(states_dir, key):
            train_xys += [xy]

    training_parameters = mlbase.TrainingParameters() \
        .epochs(epochs) \
        .batch(32)
    loss = sem.train(train_xys, training_parameters)
    del train_xys
    logging.info("Trained encoding to a final loss of: %.6f" % loss)
    sem.save(os.path.join(encoding_dir, "sem"))
    _ = score_parts(sem, lambda key: states.stream_train(states_dir, key), False)
    key_scores = score_parts(sem, lambda key: states.stream_test(states_dir, key), True)
    return key_scores


def score_parts(sem, stream_fn, output):
    key_scores = {}
    total_scores = {name: 0.0 for name in SCORES.keys()}
    count = 0

    for key in view.part_keys():
        count += 1
        scores = sem.test(stream_fn(key), False, SCORES)
        key_scores[key] = scores

        if output:
            logging.debug("Scores for '%s': %s" % (key, adjutant.dict_as_str(scores)))

        for name, score in scores.items():
            total_scores[name] += score

    user_log.info("Total scores: %s" % (adjutant.dict_as_str({name: score / float(count) for name, score in total_scores.items()})))
    return key_scores


if __name__ == "__main__":
    ret = main(sys.argv[1:])
    sys.exit(ret)

