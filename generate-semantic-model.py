
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
    "rank_score_linear": scoring.rank_score_linear(),
    "top_k1": scoring.descrete_rank(top_k=0),
    "top_k2": scoring.descrete_rank(top_k=1),
    "top_k3": scoring.descrete_rank(top_k=2),
}
moot = {
    "accuracy": scoring.accuracy,
    "rank_score_linear": scoring.rank_score_linear(),
    "rank_score_exponential": scoring.rank_score_exponential(),
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
    ap.add_argument("--word-input", default=False, action="store_true")
    ap.add_argument("-s", "--score", default=False, action="store_true")
    ap.add_argument("data_dir")
    ap.add_argument("states_dir")
    ap.add_argument("encoding_dir")
    aargs = ap.parse_args(argv)
    setup_logging(".%s.log" % os.path.splitext(os.path.basename(__file__))[0], aargs.verbose, False, True, True)
    logging.debug(aargs)

    user_log.info("Sem")
    sem = generate_sem(aargs.data_dir, aargs.layers, aargs.width, aargs.states_dir, aargs.epochs, aargs.encoding_dir, aargs.word_input)

    if aargs.score:
        scores_sem, totals_sem = test_model(sem, aargs.states_dir, False)
        user_log.info("Baseline")
        baseline = generate_baseline(aargs.data_dir, aargs.word_input)
        scores_baseline, totals_baseline = test_model(baseline, aargs.states_dir, True)

        with open(os.path.join(aargs.encoding_dir, "analysis-breakdown.csv"), "w") as fh:
            writer = csv_writer(fh)
            writer.writerow(["technique", "key", "score_fn", "result"])

            for key, scores in sorted(scores_sem.items()):
                for name, score in sorted(scores.items()):
                    writer.writerow(["sem", key, name, "%f" % score])

            for key, scores in sorted(scores_baseline.items()):
                for name, score in sorted(scores.items()):
                    writer.writerow(["baseline", key, name, "%f" % score])

        with open(os.path.join(aargs.encoding_dir, "analysis-totals.csv"), "w") as fh:
            writer = csv_writer(fh)
            writer.writerow(["technique", "score_fn", "result"])

            for key, scores in sorted(totals_sem.items()):
                writer.writerow(["sem", name, "%f" % score])

            for name, scores in sorted(totals_baseline.items()):
                writer.writerow(["baseline", name, "%f" % score])

    return 0


def generate_sem(data_dir, layers, width, states_dir, epochs, encoding_dir, word_input):
    hyper_parameters = model.HyperParameters() \
        .layers(layers) \
        .width(width)
    sem = semantic.model_for(data_dir, lambda s, i, o, e: model.Ffnn(s, i, o, hyper_parameters, e), word_input)
    train_xys = []

    for key in view.keys():
        for xy in states.stream_hidden_train(states_dir, key, _data_converter(key, word_input)):
            train_xys += [xy]

    training_parameters = mlbase.TrainingParameters() \
        .epochs(epochs) \
        .batch(32)
    loss = sem.train(train_xys, training_parameters)
    del train_xys
    logging.info("Trained semantic encoding to a final loss of: %.6f" % loss)
    semantic.save_model(sem, encoding_dir)
    return sem


def generate_baseline(data_dir, word_input):
    output_distribution = data.get_output_distribution(data_dir)

    def model_fn(s, i, o, e):
        custom_distribution = [0] * len(o)

        for key, value in output_distribution.items():
            custom_distribution[o.encode(key)] = value

        return model.CustomOutput(s, i, o, np.array(custom_distribution), e)

    return semantic.model_for(data_dir, model_fn, word_input)


def test_model(model, states_dir, is_baseline):
    word_input = model.extra["word_input"]
    #user_log.info("Train data.")
    #_, _ = score_parts(model, lambda key: states.stream_hidden_train(states_dir, key, _data_converter(key, word_input)), False, is_baseline)
    user_log.info("Test data.")
    key_scores, total_scores = score_parts(model, lambda key: states.stream_hidden_test(states_dir, key, _data_converter(key, word_input)), True, is_baseline)
    return key_scores, total_scores


def score_parts(model, stream_fn, debug, is_baseline):
    key_scores = {}
    total_scores = {name: 0.0 for name in SCORES.keys()}
    total_scores["loss"] = 0.0
    count = 0

    for key in view.keys():
        if is_baseline and count == 1:
            # We don't need to run across all the keys for the baseline - they would all be the same.
            break

        count += 1
        scores = model.test(stream_fn(key), False, SCORES, include_loss=not is_baseline)
        key_scores[key] = scores

        if debug:
            logging.debug("Scores for '%s': %s" % (key, adjutant.dict_as_str(scores)))

        for name, score in scores.items():
            total_scores[name] += score

    total_scores = {name: score / float(count) for name, score in total_scores.items()}
    user_log.info("Total scores: %s" % (adjutant.dict_as_str(total_scores)))
    return key_scores, total_scores


def _data_converter(key, word_input):
    if word_input:
        return lambda data: mlbase.Xy(semantic.as_input(key, data[0], data[1]), data[2])
    else:
        return lambda data: mlbase.Xy(semantic.as_input(key, None, data[1]), data[2])


if __name__ == "__main__":
    ret = main(sys.argv[1:])
    sys.exit(ret)

