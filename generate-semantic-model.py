
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
from ml import model
from ml import scoring
from nnwd import data
from nnwd.domain import NeuralNetwork
from nnwd import parameters
from nnwd import pickler
from nnwd import reduction
from nnwd import rnn
from nnwd import semantic
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
    ap = ArgumentParser(prog="generate-semantic-model")
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

    sem = generate_sem(aargs.data_dir, aargs.layers, aargs.width, aargs.states_dir, aargs.epochs, aargs.encoding_dir)
    baseline = generate_baseline(aargs.data_dir)

    user_log.info("Sem")
    scores_sem = test_model(sem, aargs.states_dir)
    user_log.info("Baseline")
    scores_baseline = test_model(baseline, aargs.states_dir)

    with open(os.path.join(aargs.encoding_dir, "analysis.csv"), "w") as fh:
        writer = csv_writer(fh)
        writer.writerow(["technique", "key", "score_fn", "result"])

        for key, scores in sorted(scores_sem.items()):
            for name, score in sorted(scores.items()):
                writer.writerow(["sem", key, name, "%f" % score])

        for key, scores in sorted(scores_baseline.items()):
            for name, score in sorted(scores.items()):
                writer.writerow(["baseline", key, name, "%f" % score])

    return 0


def generate_sem(data_dir, layers, width, states_dir, epochs, encoding_dir):
    hyper_parameters = model.HyperParameters() \
        .layers(layers) \
        .width(width)
    sem = semantic.model_for(data_dir, lambda s, i, o: model.Ffnn(s, i, o, hyper_parameters))
    train_xys = []

    for key in view.keys():
        for xy in states.stream_train(states_dir, key, _data_converter(key)):
            train_xys += [xy]

    training_parameters = mlbase.TrainingParameters() \
        .epochs(epochs) \
        .batch(32)
    loss = sem.train(train_xys, training_parameters)
    del train_xys
    logging.info("Trained semantic encoding to a final loss of: %.6f" % loss)
    semantic.save_model(sem, encoding_dir)
    return sem


def generate_baseline(data_dir):
    output_distribution = data.get_output_distribution(data_dir)

    def model_fn(s, i, o):
        custom_distribution = [0] * len(o)

        for key, value in output_distribution.items():
            custom_distribution[o.encode(key)] = value

        return model.CustomOutput(s, i, o, np.array(custom_distribution))

    return semantic.model_for(data_dir, model_fn)


def test_model(model, states_dir):
    user_log.info("Train data.")
    _ = score_parts(model, lambda key: states.stream_train(states_dir, key, _data_converter(key)), False)
    user_log.info("Test data.")
    key_scores = score_parts(model, lambda key: states.stream_test(states_dir, key, _data_converter(key)), True)
    return key_scores


def score_parts(model, stream_fn, debug):
    key_scores = {}
    total_scores = {name: 0.0 for name in SCORES.keys()}
    count = 0

    for key in view.keys():
        count += 1
        scores = model.test(stream_fn(key), False, SCORES)
        key_scores[key] = scores

        if debug:
            logging.debug("Scores for '%s': %s" % (key, adjutant.dict_as_str(scores)))

        for name, score in scores.items():
            total_scores[name] += score

    user_log.info("Total scores: %s" % (adjutant.dict_as_str({name: score / float(count) for name, score in total_scores.items()})))
    return key_scores


def _data_converter(key):
    return lambda data: mlbase.Xy(semantic.as_input(key, data[0]), data[1])


if __name__ == "__main__":
    ret = main(sys.argv[1:])
    sys.exit(ret)

