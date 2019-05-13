
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
from nnwd import states
from nnwd import sequential
from nnwd import view

from pytils import adjutant
from pytils.log import setup_logging, user_log


def main(argv):
    ap = ArgumentParser(prog="generate-sequential-model")
    ap.add_argument("--verbose", "-v", default=False, action="store_true", help="Turn on verbose logging.")
    #ap.add_argument("-d", "--dry-run", default=False, action="store_true")
    ap.add_argument("-b", "--batch", default=32, type=int)
    ap.add_argument("-a", "--arc-epochs", default=5, type=int)
    ap.add_argument("-c", "--consecutive-decays", default=5, type=int)
    ap.add_argument("data_dir")
    ap.add_argument("sequential_dir")
    aargs = ap.parse_args(argv)
    setup_logging(".%s.log" % os.path.splitext(os.path.basename(__file__))[0], aargs.verbose, False, True, True)
    logging.debug(aargs)
    rnn = generate_rnn(aargs.data_dir, aargs.batch, aargs.arc_epochs, aargs.consecutive_decays, aargs.sequential_dir)
    return 0


def generate_rnn(data_dir, batch, arc_epochs, consecutive_decays, sequential_dir):
    rnn = sequential.model_for(data_dir)
    train_xys = [xy for xy in data.stream_train(data_dir)]
    validation_xys = [xy for xy in data.stream_validation(data_dir)]
    test_xys = [xy for xy in data.stream_test(data_dir)]
    logging.debug("data sets (train, validation, test): %d, %d, %d" % (len(train_xys), len(validation_xys), len(test_xys)))
    converging_train(rnn, batch, arc_epochs, consecutive_decays, sequential_dir, train_xys, validation_xys, test_xys)
    del train_xys
    return rnn


def converging_train(rnn, batch, arc_epochs, consecutive_decays, sequential_dir, train_xys, validation_xys, test_xys):
    best_score_train = rnn.test(train_xys)
    best_score_validation = rnn.test(validation_xys)
    score_test = rnn.test(test_xys)
    logging.debug("Baseline train/validation/test scores (random initialized weights): %.4f / %.4f / %.4f" % (best_score_train, best_score_validation, score_test))
    training_parameters = mlbase.TrainingParameters() \
        .batch(batch) \
        .epochs(arc_epochs) \
        .convergence(False) \
        .debug(True) \
        .score(True)
    previous_loss = None
    arc = -1
    version = 0
    sequential.save_model(rnn, sequential_dir, version)
    converged = False
    decays = 0

    while not converged:
        arc += 1
        logging.debug("train lstm arc %d: %s" % (arc, training_parameters))
        loss, score_train, score_validation = rnn_train_loop(rnn, train_xys, validation_xys, training_parameters)
        loss_change = _change(previous_loss, loss, lambda prev, curr: prev > curr)
        train_change = _change(best_score_train, score_train, lambda prev, curr: prev < curr)
        validation_change = _change(best_score_validation, score_validation, lambda prev, curr: prev < curr)
        logging.debug("train lstm arc %d: (loss, tr, va) (%s %.4f, %s %.4f, %s %.4f)" % (arc, loss_change, loss, train_change, score_train, validation_change, score_validation))
        both_improved = score_train > best_score_train and score_validation > best_score_validation

        if score_train > best_score_train or score_validation > best_score_validation:
            previous_loss = loss
            version += 1
            sequential.save_model(rnn, sequential_dir, version)

            # At least one improved.
            if score_train > best_score_train:
                best_score_train = score_train

            if score_validation > best_score_validation:
                best_score_validation = score_validation
            else:
                # The validation score didn't improve.  Lets see where the test score is at.
                score_test = rnn.test(test_xys)
                logging.debug("test score: %.4f" % score_test)
        else:
            # Neither improved.
            # Load the best known version to continue training off of.
            sequential.load_model(rnn, sequential_dir)

        if not both_improved:
            if decays > consecutive_decays:
                converged = True

            logging.debug("decaying..")
            training_parameters = training_parameters.decay()
            decays += 1
        else:
            decays = 0

    # Load which ever version was marked as the latest as the final trained lstm.
    sequential.load_model(rnn, sequential_dir)
    logging.debug("Calculating final scores.")
    score_train = rnn.test(train_xys, False)
    score_validation = rnn.test(validation_xys, False)
    score_test = rnn.test(test_xys, True)
    logging.debug("(tr, va, te): (%.4f, %.4f, %.4f)" % (score_train, score_validation, score_test))


def _change(previous, current, better_fn):
    if previous is None:
        return "-"
    elif better_fn(previous, current):
        return "▲"
    else:
        return "▼"


def rnn_train_loop(rnn, train_xys, validation_xys, training_parameters):
    loss, score_train = rnn.train(train_xys, training_parameters)
    score_validation = rnn.test(validation_xys)
    return loss, score_train, score_validation


if __name__ == "__main__":
    ret = main(sys.argv[1:])
    sys.exit(ret)
