#!/usr/bin/python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import logging
import os
import sys

from nnwd import domain
from nnwd import rnn
from pytils.log import setup_logging, user_log


def main(argv):
    ap = ArgumentParser(prog="language-model")
    ap.add_argument("--verbose", "-v",
                    default=False,
                    action="store_true",
                    help="Turn on verbose logging.")
    ap.add_argument("--corpus", default="corpus.txt")
    ap.add_argument("--epochs", default=100, type=int)
    args = ap.parse_args(argv)
    setup_logging(".%s.log" % os.path.splitext(os.path.basename(__file__))[0], args.verbose, False, True)
    words, xy_sequences, neural_network = domain.create(args.corpus, args.epochs, args.verbose)

    #while neural_network.is_setting_up():
    #    pass

    neural_network._background_training.join()
    accuracy = neural_network.lstm.test([[rnn.Xy(t[0], t[1]) for t in sequence] for sequence in xy_sequences], True)
    user_log.info("accuracy: %s" % accuracy)


if __name__ == "__main__":
    sys.exit(main())

