
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
from nnwd import parameters
from nnwd import pickler
from nnwd import reduction
from nnwd import semantic
from nnwd import sequential
from nnwd import states

from pytils import adjutant
from pytils.log import setup_logging, teardown, user_log


@teardown
def main(argv):
    ap = ArgumentParser(prog="generate-semantic-model")
    ap.add_argument("-v", "--verbose", default=False, action="store_true", help="Turn on verbose logging.")
    ap.add_argument("-i", "--initial-decays", default=5, type=int)
    ap.add_argument("-c", "--convergence-decays", default=2, type=int)
    ap.add_argument("-a", "--arc-epochs", default=3, type=int)
    ap.add_argument("-l", "--layers", default=2, type=int)
    ap.add_argument("-w", "--width", default=100, type=int)
    ap.add_argument("--word-input", default=False, action="store_true")
    ap.add_argument("-p", "--pre-existing", default=False, action="store_true")
    ap.add_argument("-m", "--monolith", default=False, action="store_true")
    ap.add_argument("--key-set", nargs="*", default=None)
    ap.add_argument("data_dir")
    ap.add_argument("sequential_dir")
    ap.add_argument("states_dir")
    ap.add_argument("encoding_dir")
    aargs = ap.parse_args(argv)
    setup_logging(".%s.log" % os.path.splitext(os.path.basename(__file__))[0], aargs.verbose, False, True, True)
    logging.debug(aargs)

    lstm = sequential.load_model(aargs.data_dir, aargs.sequential_dir, True)
    user_log.info("Sem")
    hyper_parameters = model.HyperParameters(aargs.layers, aargs.width)
    extra = {
        "word_input": aargs.word_input,
        "monolith": aargs.monolith,
    }

    if aargs.pre_existing:
        sem = load_sem(lstm, aargs.encoding_dir)
    else:
        sem = generate_sem(lstm, hyper_parameters, extra, aargs.states_dir, aargs.arc_epochs, aargs.encoding_dir, aargs.key_set, aargs.initial_decays, aargs.convergence_decays)

    keys_sem, total_sem = test_model(lstm, sem, aargs.states_dir, False, aargs.key_set)
    # TODO
    #user_log.info("Baseline")
    #baseline = generate_baseline(aargs.data_dir, lstm, hyper_parameters, extra)
    #scores_baseline, totals_baseline = test_model(lstm, baseline, aargs.states_dir, True, aargs.key_set)

    with open(os.path.join(aargs.encoding_dir, "analysis-breakdown.csv"), "w") as fh:
        writer = csv_writer(fh)
        writer.writerow(["technique", "key", "perplexity"])

        for key, perplexity in sorted(keys_sem.items()):
            writer.writerow(["sem", key, "%f" % perplexity])

        #for key, scores in sorted(scores_baseline.items()):
        #    for name, score in sorted(scores.items()):
        #        writer.writerow(["baseline", key, name, "%f" % score])

    with open(os.path.join(aargs.encoding_dir, "analysis-totals.csv"), "w") as fh:
        writer = csv_writer(fh)
        writer.writerow(["technique", "perplexity"])
        writer.writerow(["sem", "%f" % total_sem])

        #for name, score in sorted(totals_baseline.items()):
        #    writer.writerow(["baseline", name, "%f" % score])

    return 0


def _ffnn_constructor(scope, hyper_parameters, extra, case_field, hidden_vector, word_labels, output_labels):
    if extra["word_input"]:
        input_field = mlbase.ConcatField([case_field, hidden_vector, word_labels])
    else:
        input_field = mlbase.ConcatField([case_field, hidden_vector])

    if extra["monolith"]:
        return model.Ffnn(scope, hyper_parameters, extra, input_field, output_labels)
    else:
        return model.SeparateFfnn(scope, hyper_parameters, extra, input_field, output_labels, case_field)


def load_sem(lstm, encoding_dir):
    return semantic.load_model(lstm, encoding_dir, model_fn=_ffnn_constructor)


def generate_sem(lstm, hyper_parameters, extra, states_dir, arc_epochs, encoding_dir, key_set, initial_decays, convergence_decays):
    sem = semantic.model_for(lstm, hyper_parameters=hyper_parameters, extra=extra, model_fn=_ffnn_constructor)
    as_input = as_input_fn(lstm, sem)

    def train_xys():
        for key, hidden_state in states.random_stream_hidden_states(states_dir, "train", key_set):
            yield mlbase.Xy(as_input(key, hidden_state), hidden_state.annotation)

    def validation_xys():
        for key, hidden_state in states.random_stream_hidden_states(states_dir, "validation", key_set):
            yield mlbase.Xy(as_input(key, hidden_state), hidden_state.annotation)

    def test_xys():
        for key, hidden_state in states.random_stream_hidden_states(states_dir, "test", key_set):
            yield mlbase.Xy(as_input(key, hidden_state), hidden_state.annotation)

    semantic.train_model(sem, [train_xys, validation_xys, test_xys], encoding_dir, arc_epochs, initial_decays, convergence_decays)
    return sem


def generate_baseline(data_dir, lstm, hyper_parameters, extra):
    output_distribution = data.get_output_distribution(data_dir)

    def custom_model(scope, hp, e, c, h, w, output_labels):
        custom_distribution = [0] * len(output_labels)

        for key, value in output_distribution.items():
            custom_distribution[output_labels.encode(key)] = value

        return model.CustomOutput(scope, output_labels, np.array(custom_distribution))

    return semantic.model_for(lstm, hyper_parameters=hyper_parameters, extra=extra, model_fn=custom_model)


def test_model(lstm, model, states_dir, is_baseline, key_set):
    as_input = as_input_fn(lstm, model)

    def stream_fn(key):
        for hidden_state in states.stream_hidden_test(states_dir, key):
            yield mlbase.Xy(as_input(key, hidden_state), hidden_state.annotation)

    #user_log.info("Train data.")
    #_, _ = score_parts(model, stream_fn, False, is_baseline)
    user_log.info("Test data.")
    key_perplexity, total_perplexity = score_parts(lstm, model, stream_fn, True, is_baseline, key_set)
    return key_perplexity, total_perplexity


def score_parts(lstm, model, stream_fn, debug, is_baseline, key_set):
    key_perplexity = {}
    total_perplexity = 0.0
    count = 0

    for key in lstm.keys():
        if key_set is None or key in key_set:
            if is_baseline and count == 1:
                # We don't need to run across all the keys for the baseline - they would all be the same.
                break

            count += 1
            perplexity = model.test(lambda: stream_fn(key), False)
            key_perplexity[key] = perplexity

            if debug:
                logging.debug("Perplexity for '%s': %.6f" % (key, perplexity))

            total_perplexity += perplexity

    total_perplexity = total_perplexity / count
    user_log.info("Total perplexity: %.6f" % total_perplexity)
    return key_perplexity, total_perplexity


def as_input_fn(lstm, model):
    embedding_padding = tuple([0] * max(0, lstm.hyper_parameters.width - lstm.hyper_parameters.embedding_width))
    hidden_padding = tuple([0] * max(0, lstm.hyper_parameters.embedding_width - lstm.hyper_parameters.width))

    if hasattr(model, "extra") and model.extra["word_input"]:
        def converter(key, hidden_state):
            return (key, tuple(hidden_state.point) + (embedding_padding if lstm.is_embedding(key) else hidden_padding), hidden_state.word)
    else:
        def converter(key, hidden_state):
            return (key, tuple(hidden_state.point) + (embedding_padding if lstm.is_embedding(key) else hidden_padding))

    return converter


if __name__ == "__main__":
    ret = main(sys.argv[1:])
    sys.exit(ret)

