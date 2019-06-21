
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
from nnwd import pickler
from nnwd import reduction
from nnwd import rnn
from nnwd import states
from nnwd import sequential

from pytils import adjutant
from pytils.log import setup_logging, teardown, user_log


@teardown
def main(argv):
    ap = ArgumentParser(prog="generate-sequential-model")
    ap.add_argument("-v", "--verbose", default=False, action="store_true", help="Turn on verbose logging.")
    #ap.add_argument("-d", "--dry-run", default=False, action="store_true")
    ap.add_argument("-l", "--layers", default=2, type=int)
    ap.add_argument("-w", "--width", default=100, type=int)
    ap.add_argument("-e", "--embedding-width", default=50, type=int)
    ap.add_argument("--srnn", default=False, action="store_true", help="use the 'srnn' ablation")
    ap.add_argument("--out", default=False, action="store_true", help="use the 'out' ablation")
    ap.add_argument("-b", "--batch", default=32, type=int)
    ap.add_argument("-a", "--arc-epochs", default=5, type=int)
    ap.add_argument("-i", "--initial-decays", default=5, type=int)
    ap.add_argument("-c", "--convergence-decays", default=2, type=int)
    ap.add_argument("data_dir")
    ap.add_argument("sequential_dir")
    aargs = ap.parse_args(argv)
    setup_logging(".%s.log" % os.path.splitext(os.path.basename(__file__))[0], aargs.verbose, False, True, True)
    logging.debug(aargs)
    hyper_parameters = sequential.HyperParameters(aargs.layers, aargs.width, aargs.embedding_width)
    ablations = sequential.Ablations(aargs.srnn, aargs.out)
    rnn = generate_rnn(aargs.data_dir, hyper_parameters, ablations, aargs.batch, aargs.arc_epochs, aargs.initial_decays, aargs.convergence_decays, aargs.sequential_dir)
    return 0


def generate_rnn(data_dir, hyper_parameters, ablations, batch, arc_epochs, initial_decays, convergence_decays, sequential_dir):
    rnn = sequential.model_for(data_dir, hyper_parameters=hyper_parameters, ablations=ablations)
    train_xys = [xy for xy in data.stream_train(data_dir)]
    validation_xys = [xy for xy in data.stream_validation(data_dir)]
    test_xys = [xy for xy in data.stream_test(data_dir)]
    logging.debug("data sets (train, validation, test): %d, %d, %d" % (len(train_xys), len(validation_xys), len(test_xys)))
    show_score = data.get_description(data_dir).task == data.SA
    converging_train(rnn, batch, arc_epochs, initial_decays, convergence_decays, sequential_dir, train_xys, validation_xys, test_xys, show_score)
    del train_xys
    return rnn


def converging_train(rnn, batch, arc_epochs, initial_decays, convergence_decays, sequential_dir, train_xys, validation_xys, test_xys, show_score):
    assert initial_decays > convergence_decays, "%d <= %d" % (initial_decays, convergence_decays)
    best_score_train = rnn.test(train_xys, score=show_score)
    best_score_validation = rnn.test(validation_xys, score=show_score)
    score_test = rnn.test(test_xys, score=show_score)
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
    initialized = False
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
            initialized = True
            sequential.save_parameters(rnn, sequential_dir, version)

            # At least one improved.
            if score_train > best_score_train:
                best_score_train = score_train

            if score_validation > best_score_validation:
                best_score_validation = score_validation
            else:
                # The validation score didn't improve.  Lets see where the test score is at.
                score_test = rnn.test(test_xys, score=show_score)
                logging.debug("test score: %.4f" % score_test)
        else:
            # Neither improved.
            # Load the best known version to continue training off of.
            sequential.load_parameters(rnn, sequential_dir)

        if not both_improved:
            if decays >= convergence_decays if initialized else decays >= initial_decays:
                converged = True
                logging.debug("Converged" + ("" if initialized else " without initialization!"))
            else:
                decays += 1
                logging.debug("Decaying.. %d", decays)
                training_parameters = training_parameters.decay(initial=initialized)
        else:
            logging.debug("Reset decay")
            decays = 0

    # Load which ever version was marked as the latest as the final trained lstm.
    sequential.load_parameters(rnn, sequential_dir)
    logging.debug("Calculating final scores.")
    score_train = rnn.test(train_xys, False, score=show_score)
    score_validation = rnn.test(validation_xys, False, score=show_score)
    score_test = rnn.test(test_xys, True, score=show_score)
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

