
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
from nnwd import rnn
from nnwd import semantic
from nnwd import sequential
from nnwd import states

from pytils import adjutant
from pytils.log import setup_logging, teardown, user_log


@teardown
def main(argv):
    ap = ArgumentParser(prog="generate-semantic-model")
    ap.add_argument("-v", "--verbose", default=False, action="store_true", help="Turn on verbose logging.")
    ap.add_argument("-e", "--epochs", default=10, type=int)
    ap.add_argument("-l", "--layers", default=2, type=int)
    ap.add_argument("-w", "--width", default=100, type=int)
    ap.add_argument("--word-input", default=False, action="store_true")
    ap.add_argument("-p", "--pre-existing", default=False, action="store_true")
    ap.add_argument("--key-set", nargs="*", default=None)
    ap.add_argument("-m", "--monolith", default=False, action="store_true")
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
    extra = {"word_input": aargs.word_input}
    model_fn = _ffnn_constructor if aargs.monolith else _switch_ffnn_constructor

    if aargs.pre_existing:
        sem = load_sem(lstm, aargs.encoding_dir, model_fn)
    else:
        sem = generate_sem(lstm, hyper_parameters, extra, aargs.states_dir, aargs.epochs, aargs.encoding_dir, model_fn, aargs.key_set, aargs.monolith)

    scores_sem, totals_sem = test_model(lstm, sem, aargs.states_dir, False, aargs.key_set)
    # TODO
    #user_log.info("Baseline")
    #baseline = generate_baseline(aargs.data_dir, lstm, hyper_parameters, extra)
    #scores_baseline, totals_baseline = test_model(lstm, baseline, aargs.states_dir, True, aargs.key_set)

    with open(os.path.join(aargs.encoding_dir, "analysis-breakdown.csv"), "w") as fh:
        writer = csv_writer(fh)
        writer.writerow(["technique", "key", "score_fn", "result"])

        for key, scores in sorted(scores_sem.items()):
            for name, score in sorted(scores.items()):
                writer.writerow(["sem", key, name, "%f" % score])

        #for key, scores in sorted(scores_baseline.items()):
        #    for name, score in sorted(scores.items()):
        #        writer.writerow(["baseline", key, name, "%f" % score])

    with open(os.path.join(aargs.encoding_dir, "analysis-totals.csv"), "w") as fh:
        writer = csv_writer(fh)
        writer.writerow(["technique", "score_fn", "result"])

        for name, score in sorted(totals_sem.items()):
            writer.writerow(["sem", name, "%f" % score])

        #for name, score in sorted(totals_baseline.items()):
        #    writer.writerow(["baseline", name, "%f" % score])

    return 0


def _ffnn_constructor(scope, hyper_parameters, extra, case_field, hidden_vector, word_labels, output_labels):
    if extra["word_input"]:
        input_field = mlbase.ConcatField([case_field, hidden_vector, word_labels])
    else:
        input_field = mlbase.ConcatField([case_field, hidden_vector])

    return model.Ffnn(scope, hyper_parameters, extra, input_field, output_labels)


def _switch_ffnn_constructor(scope, hyper_parameters, extra, case_field, hidden_vector, word_labels, output_labels):
    if extra["word_input"]:
        statement_field = mlbase.ConcatField([case_field, hidden_vector, word_labels])
    else:
        statement_field = mlbase.ConcatField([case_field, hidden_vector])

    return model.SwitchFfnn(scope, hyper_parameters, extra, case_field, statement_field, output_labels)


def load_sem(lstm, encoding_dir, model_fn):
    return semantic.load_model(lstm, encoding_dir, model_fn=model_fn)


def generate_sem(lstm, hyper_parameters, extra, states_dir, epochs, encoding_dir, model_fn, key_set, monolith):
    sem = semantic.model_for(lstm, hyper_parameters=hyper_parameters, extra=extra, model_fn=model_fn)
    semantic.save_model(sem, encoding_dir)
    as_input = as_input_fn(lstm, sem)

    if monolith:
        def train_xys():
            for key, hidden_state in states.stream_all_hidden_train(states_dir):
                if key_set is None or key in key_set:
                    yield mlbase.Xy(as_input(key, hidden_state), hidden_state.annotation)
    else:
        def training_subset(key):
            def inner():
                for hidden_state in states.stream_hidden_train(states_dir, key):
                    yield mlbase.Xy(as_input(key, hidden_state), hidden_state.annotation)

            return inner

        train_xys = {key: training_subset(key) for key in lstm.keys() if key_set is None or key in key_set}

    training_parameters = mlbase.TrainingParameters() \
        .epochs(epochs) \
        .batch(32)
    loss = sem.train(train_xys, training_parameters)
    logging.info("Trained semantic encoding to a final loss of: %.6f" % loss)
    semantic.save_parameters(sem, encoding_dir)
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
    key_scores, total_scores = score_parts(lstm, model, stream_fn, True, is_baseline, key_set)
    return key_scores, total_scores


def score_parts(lstm, model, stream_fn, debug, is_baseline, key_set):
    key_scores = {}
    total_scores = {}
    total_scores["loss"] = 0.0
    total_scores["perplexity"] = 0.0
    count = 0

    for key in lstm.keys():
        if key_set is None or key in key_set:
            if is_baseline and count == 1:
                # We don't need to run across all the keys for the baseline - they would all be the same.
                break

            count += 1
            scores = model.test(stream_fn(key), False, include_loss=True)
            key_scores[key] = scores

            if debug:
                logging.debug("Scores for '%s': %s" % (key, adjutant.dict_as_str(scores)))

            for name, score in scores.items():
                total_scores[name] += score

    total_scores = {name: score / float(count) for name, score in total_scores.items()}
    user_log.info("Total scores: %s" % (adjutant.dict_as_str(total_scores)))
    return key_scores, total_scores


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

