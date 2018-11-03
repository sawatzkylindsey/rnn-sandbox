#!/usr/bin/python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import logging
import nlp
import os
import rnn
import sys

from pytils.log import setup_logging, user_log


def main():
    ap = ArgumentParser(prog="language-model")
    ap.add_argument("--verbose", "-v",
                    default=False,
                    action="store_true",
                    help="Turn on verbose logging.")
    ap.add_argument("training_corpus")
    args = ap.parse_args()
    setup_logging(".%s.log" % os.path.splitext(os.path.basename(__file__))[0], args.verbose, False, True)
    words, xy_sequences = nlp.corpus_sequences(args.training_corpus)
    xy_sequences = [[rnn.Xy(x, y) for x, y in sequence] for sequence in xy_sequences]
    neural_network = rnn.Rnn(1, 5, words)
    neural_network.train(xy_sequences, 100, True)
    accuracy = neural_network.test(xy_sequences, True)
    user_log.info("accuracy: %s" % accuracy)


if __name__ == "__main__":
    sys.exit(main())

