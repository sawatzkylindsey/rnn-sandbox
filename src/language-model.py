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
    words, xy_sequences = load_corpus(args.training_corpus)
    neural_network = rnn.Rnn(1, 5, words)
    neural_network.train(xy_sequences, 1000, True)
    accuracy = neural_network.test(xy_sequences, True)
    user_log.info("accuracy: %s" % accuracy)


def load_corpus(training_corpus_file):
    corpus_lines = None

    with open(training_corpus_file, "r") as fh:
        corpus_lines = fh.readlines()

    words = set()
    xy_sequences = []

    for line in corpus_lines:
        for sentence in nlp.split_sentences(line):
            sequence = []

            for i, word in enumerate(sentence):
                words.add(word)

                if i + 1 < len(sentence):
                    sequence.append(rnn.Xy(word, sentence[i + 1]))

            xy_sequences.append(sequence)

    labels = nlp.Labels(words, unknown=nlp.UNKNOWN)
    logging.info("words (%d): %s" % (len(labels), labels))
    return labels, xy_sequences


if __name__ == "__main__":
    sys.exit(main())

