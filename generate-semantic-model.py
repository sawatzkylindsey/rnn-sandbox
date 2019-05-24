
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


SCORES = [
    ("top_k100", scoring.descrete_rank(top_k=100)),
    ("top_k25", scoring.descrete_rank(top_k=25)),
    ("top_k10", scoring.descrete_rank(top_k=10)),
    ("top_k1", scoring.descrete_rank(top_k=0)),
    ("top_k2", scoring.descrete_rank(top_k=1)),
    ("top_k3", scoring.descrete_rank(top_k=2)),
]


@teardown
def main(argv):
    ap = ArgumentParser(prog="generate-semantic-model")
    ap.add_argument("-v", "--verbose", default=False, action="store_true", help="Turn on verbose logging.")
    ap.add_argument("-e", "--epochs", default=10, type=int)
    ap.add_argument("-l", "--layers", default=2, type=int)
    ap.add_argument("-w", "--width", default=100, type=int)
    ap.add_argument("--word-input", default=False, action="store_true")
    ap.add_argument("-s", "--score", default=False, action="store_true")
    ap.add_argument("-p", "--pre-existing", default=False, action="store_true")
    ap.add_argument("data_dir")
    ap.add_argument("sequential_dir")
    ap.add_argument("states_dir")
    ap.add_argument("encoding_dir")
    aargs = ap.parse_args(argv)
    setup_logging(".%s.log" % os.path.splitext(os.path.basename(__file__))[0], aargs.verbose, False, True, True)
    logging.debug(aargs)

    lstm = sequential.load_model(aargs.data_dir, aargs.sequential_dir)
    user_log.info("Sem")
    hyper_parameters = model.HyperParameters(aargs.layers, aargs.width)
    extra = {"word_input": aargs.word_input}

    if aargs.pre_existing:
        sem, sem_as_input = load_sem(lstm, aargs.encoding_dir)
    else:
        sem, sem_as_input = generate_sem(lstm, hyper_parameters, extra, aargs.states_dir, aargs.epochs, aargs.encoding_dir)

    if aargs.score:
        scores_sem, totals_sem = test_model(lstm, sem, sem_as_input, aargs.states_dir, False)
        user_log.info("Baseline")
        baseline, baseline_as_input = generate_baseline(aargs.data_dir, lstm, hyper_parameters, extra)
        scores_baseline, totals_baseline = test_model(lstm, baseline, baseline_as_input, aargs.states_dir, True)

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


def load_sem(lstm, encoding_dir):
    return semantic.load_model(lstm, encoding_dir, model_fn=lambda hp, e, i, o, s: model.Ffnn(hp, e, i, o, s))


def generate_sem(lstm, hyper_parameters, extra, states_dir, epochs, encoding_dir):
    sem, as_input = semantic.model_for(lstm, hyper_parameters=hyper_parameters, extra=extra, model_fn=lambda hp, e, i, o, s: model.Ffnn(hp, e, i, o, s))
    semantic.save_model(sem, encoding_dir)
    train_xys = [mlbase.Xy(as_input(key, hidden_state), hidden_state.annotation) for key, hidden_state in states.stream_hidden_train(states_dir)]
    training_parameters = mlbase.TrainingParameters() \
        .epochs(epochs) \
        .batch(32)
    loss = sem.train(train_xys, training_parameters)
    del train_xys
    logging.info("Trained semantic encoding to a final loss of: %.6f" % loss)
    semantic.save_parameters(sem, encoding_dir)
    return sem, as_input


def generate_baseline(data_dir, lstm, hyper_parameters, extra):
    output_distribution = data.get_output_distribution(data_dir)

    def custom_model(hp, e, i, o, s):
        custom_distribution = [0] * len(o)

        for key, value in output_distribution.items():
            custom_distribution[o.encode(key)] = value

        return model.CustomOutput(hp, e, i, o, s, np.array(custom_distribution))

    return semantic.model_for(lstm, hyper_parameters=hyper_parameters, extra=extra, model_fn=custom_model)


def test_model(lstm, model, as_input, states_dir, is_baseline):
    def stream_fn(key):
        for hidden_state in states.stream_hidden_test(states_dir, key):
            yield mlbase.Xy(as_input(key, hidden_state), hidden_state.annotation)

    #user_log.info("Train data.")
    #_, _ = score_parts(model, stream_fn, False, is_baseline)
    user_log.info("Test data.")
    key_scores, total_scores = score_parts(lstm, model, stream_fn, True, is_baseline)
    return key_scores, total_scores


def score_parts(lstm, model, stream_fn, debug, is_baseline):
    key_scores = {}
    total_scores = {name: 0.0 for name, function in SCORES}
    total_scores["loss"] = 0.0
    count = 0

    for key in lstm.keys():
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


if __name__ == "__main__":
    ret = main(sys.argv[1:])
    sys.exit(ret)

