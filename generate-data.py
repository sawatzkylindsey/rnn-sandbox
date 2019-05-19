
from argparse import ArgumentParser
import bz2
import collections
from csv import writer as csv_writer
import glob
import logging
import math
import nltk
import numpy as np
import os
import pdb
import queue
import random
import re
from sklearn.mixture import GaussianMixture
import sys

from ml import base as mlbase
from ml import nlp
from ml import scoring
from nnwd import data
from nnwd import lm
from nnwd import pickler
from nnwd import reduction
from nnwd import rnn
from nnwd import sa
from nnwd import semantic
from nnwd import states

from pytils import adjutant
from pytils.log import setup_logging, teardown, user_log


@teardown
def main(argv):
    ap = ArgumentParser(prog="generate-data")
    ap.add_argument("-v", "--verbose", default=False, action="store_true", help="Turn on verbose logging.")
    #ap.add_argument("-d", "--dry-run", default=False, action="store_true")
    ap.add_argument("task", help="Either 'sa' or 'lm'.")
    ap.add_argument("form", help="How the language data should be interpreted:\n" \
                                 "raw: the text is raw (must still be run through a tokenizer)." \
                                 "tokenized: the text has been tokenized (space separate tokens, new lines separate sentences)." \
                                 "ptb: the text is tokenized and pos tagged in Penn Treebank form.")
    ap.add_argument("corpus_paths", nargs="+")
    ap.add_argument("data_dir")
    aargs = ap.parse_args(argv)
    setup_logging(".%s.log" % os.path.splitext(os.path.basename(__file__))[0], aargs.verbose, False, True, True)
    logging.debug(aargs)

    if aargs.task == "sa":
        assert aargs.form == "tokenized"
        train_xys, validation_xys, test_xys = sa.create(aargs.data_dir, lambda: stream_input_stanford(aargs.corpus_paths[0]))
    elif aargs.task == "lm":
        train_xys, validation_xys, test_xys = lm.create(aargs.data_dir, lambda: stream_input_text(aargs.corpus_paths, aargs.form))
    else:
        raise ValueError("Unknown task: %s" % aargs.task)

    logging.debug("data sets (train, validation, test): %d, %d, %d" % (len(train_xys), len(validation_xys), len(test_xys)))
    return 0


POS_MAP = {
    "CC": "CC",
    "CD": "CD",
    "DT": "DT",
    "EX": "EX",
    "FW": "FW",
    "IN": "IN",
    "JJ": "JJ",
    "JJR": "JJR",
    "JJS": "JJS",
    "LS": "LS",
    "MD": "MD",
    "NN": "NN",
    "NNS": "NNS",
    "NNP": "NNP",
    "NNPS": "NNPS",
    "PDT": "PDT",
    "POS": "POS",
    "PRP": "PRP",
    "PRP$": "PRP$",
    "RB": "RB",
    "RBR": "RBR",
    "RBS": "RBS",
    "RP": "RP",
    "SYM": "SYM",
    "TO": "TO",
    "UH": "UH",
    "VB": "VB",
    "VBD": "VBD",
    "VBG": "VBG",
    "VBN": "VBN",
    "VBP": "VBP",
    "VBZ": "VBZ",
    "WDT": "WDT",
    "WP": "WP",
    "WP$": "WP$",
    "WRB": "WRB",
    ".": "PUNCT",
    ",": "PUNCT",
    "``": "PUNCT",
    "''": "PUNCT",
    ":": "PUNCT",
    ";": "PUNCT",
    "(": "PUNCT",
    ")": "PUNCT",
    "$": "PUNCT",
    "!": "PUNCT",
    "?": "PUNCT",
}
BAD_TAGS = {}


def stream_input_text(input_files, form):
    for input_file in input_files:
        opener = lambda: open(input_file, "r") if not input_file.endswith("bz2") else bz2.BZ2File(input_file)

        with opener() as fh:
            for line in fh.readlines():
                if isinstance(line, bytes):
                    line = line.decode("utf-8")

                if line.strip() != "":
                    if form == "raw":
                        for sentence in nlp.split_sentences(line):
                            tagged = nltk.pos_tag(sentence)
                            sequence = []

                            for item in tagged:
                                word, tag = item

                                if tag in POS_MAP:
                                    word = word if tag != "CD" else nlp.NUMBER
                                    pos = POS_MAP[tag]
                                    sequence += [(word, pos)]
                                elif tag not in BAD_TAGS:
                                    BAD_TAGS[tag] = None
                                    print(tag)

                            yield sequence
                    elif form == "tokenized":
                        yield [(word, None) for word in line.split(" ")]
                    elif form == "ptb":
                        sequence = []

                        for item in re.split("[()]", line):
                            pair = item.strip().split(" ")

                            if len(pair) == 2:
                                tag = pair[0]

                                if tag in POS_MAP:
                                    word = pair[1].lower() if tag != "CD" else nlp.NUMBER
                                    pos = POS_MAP[tag]
                                    sequence += [(word, pos)]
                                elif tag not in BAD_TAGS:
                                    BAD_TAGS[tag] = None
                                    print(tag)

                        yield sequence
                    else:
                        raise ValueError("invalid form '%s'" % form)


def stream_input_stanford(stanford_folder):
    sentiments = {}

    with open(os.path.join(stanford_folder, "sentiment_labels.txt"), "r") as fh:
        first = True

        for line in fh.readlines():
            if first:
                first = False
            else:
                phrase_id, sentiment = line.split("|")
                sentiment = sentiment.strip()
                sentiments[phrase_id] = float(sentiment)

    dictionary = {}

    with open(os.path.join(stanford_folder, "dictionary.txt"), "r") as fh:
        for line in fh.readlines():
            line = line.strip()
            index = line.rindex("|")
            phrase = line[:index].lower()
            dictionary[phrase] = sentiments[line[index + 1:]]
            dictionary[phrase + " ."] = sentiments[line[index + 1:]]

    dataset_splits = {}

    with open(os.path.join(stanford_folder, "datasetSplit.txt"), "r") as fh:
        first = True

        for line in fh.readlines():
            if first:
                first = False
            else:
                sentence_id, label = line.split(",")
                label = label.strip()
                dataset_splits[sentence_id] = "train" if label == "1" else ("test" if label == "2" else "dev")

    #train_sentences = []

    with open(os.path.join(stanford_folder, "datasetSentences.txt"), "r") as fh:
        first = True

        for line in fh.readlines():
            if first:
                first = False
            else:
                sentence_id, sentence = line.split("\t")
                sentence = sentence.strip().lower()
                sequence = []

                for word in sentence.split(" "):
                    sequence += [word]

                #if dataset_splits[sentence_id] == "train":
                #    train_sentences += [sequence]

                yield (dataset_splits[sentence_id], sequence, dictionary[sentence])

    #data_tenth = max(1, int(len(dictionary) / 10.0))

    #for i, phrase_sentiment in enumerate(dictionary.items()):
    #    if i % data_tenth == 0 or i + 1 == len(dictionary):
    #        print("%d%% through" % int((i + 1) * 100 / len(dictionary)))

    #    # Sample at 30% rate.
    #    if random.randint(0, 9) < 3:
    #        phrase, sentiment = phrase_sentiment

    #        if any([phrase in sentence for sentence in train_sentences]):
    #            yield ("train", phrase.split(" "), sentiment)


if __name__ == "__main__":
    ret = main(sys.argv[1:])
    sys.exit(ret)

