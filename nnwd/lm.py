
import logging
import pdb
import random

from nnwd import data
from pytils import adjutant


def create(data_dir, corpus_stream_fn):
    xys = [sentence for sentence in corpus_stream_fn()]
    random.shuffle(xys)
    split_1 = int(len(xys) * 0.8)
    split_2 = split_1 + int(len(xys) * 0.1)
    train_xys = xys[:split_1]
    validation_xys = xys[split_1:split_2]
    test_xys = xys[split_2:]
    ## Words are actually only the subset from the training data.
    #words = set(adjutant.flat_map([[word_pos[0] for word_pos in sentence] for sentence in train_xys]))
    pos_tags = set(adjutant.flat_map([[word_pos[1] for word_pos in sentence] for sentence in train_xys]))
    word_pos_counts = {}

    for sentence in train_xys:
        for word_pos in sentence:
            if word_pos[0] not in word_pos_counts:
                word_pos_counts[word_pos[0]] = {}

            if word_pos[1] not in word_pos_counts[word_pos[0]]:
                word_pos_counts[word_pos[0]][word_pos[1]] = 0

            word_pos_counts[word_pos[0]][word_pos[1]] += 1

    total = 0
    word_pos_counts2 = {}

    for word, counts in word_pos_counts.items():
        summed = sum(counts.values())

        if summed > data.MINIMUM_OCCURRENCE_COUNT:
            word_pos_counts2[word] = counts
            total += summed

    output_distribution = {}
    pos_mapping = {}

    for word, counts in word_pos_counts2.items():
        output_distribution[word] = sum(counts.values()) / float(total)
        pos_count = sorted(counts.items(), key=lambda item: item[1], reverse=True)[0]
        pos_mapping[word] = pos_count[0]

    data.set_train(data_dir, train_xys)
    data.set_validation(data_dir, validation_xys)
    data.set_test(data_dir, test_xys)
    data.set_output_distribution(data_dir, output_distribution)
    words = set([word for word in word_pos_counts2.keys()])
    data.set_words(data_dir, words)
    data.set_pos_mapping(data_dir, pos_mapping)
    data.set_pos(data_dir, pos_tags)
    data.set_description(data_dir, data.Description(data.LM))
    logging.debug("total pairs (t, v, t): %d, %d, %d" % (sum([len(xy) for xy in train_xys]), sum([len(xy) for xy in validation_xys]), sum([len(xy) for xy in test_xys])))
    logging.debug("unique words: %d" % len(words))
    return train_xys, validation_xys, test_xys


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

COARSE_MAP = {
    "CC": "OTHER",
    "CD": "OTHER",
    "DT": "OTHER",
    "EX": "NOUN",
    "FW": "OTHER",
    "IN": "OTHER",
    "JJ": "ADJ",
    "JJR": "ADJ",
    "JJS": "ADJ",
    "LS": "OTHER",
    "MD": "VERB",
    "NN": "NOUN",
    "NNS": "NOUN",
    "NNP": "NOUN",
    "NNPS": "NOUN",
    "PDT": "OTHER",
    "POS": "OTHER",
    "PRP": "NOUN",
    "PRP$": "NOUN",
    "RB": "ADV",
    "RBR": "ADV",
    "RBS": "ADV",
    "RP": "ADV",
    "SYM": "OTHER",
    "TO": "OTHER",
    "UH": "OTHER",
    "VB": "VERB",
    "VBD": "VERB",
    "VBG": "VERB",
    "VBN": "VERB",
    "VBP": "VERB",
    "VBZ": "VERB",
    "WDT": "OTHER",
    "WP": "NOUN",
    "WP$": "NOUN",
    "WRB": "ADV",
    "PUNCT": "OTHER",
}

UNIVSERAL_MAP = {
    "CC": "CONJ",
    "CD": "NUM",
    "DT": "DET",
    "EX": "PRON",
    "FW": "X",
    "IN": "ADP",
    "JJ": "ADJ",
    "JJR": "ADJ",
    "JJS": "ADJ",
    "LS": "X",
    "MD": "VERB",
    "NN": "NOUN",
    "NNS": "NOUN",
    "NNP": "NOUN",
    "NNPS": "NOUN",
    "PDT": "DET",
    "POS": "PART",
    "PRP": "PRON",
    "PRP$": "DET",
    "RB": "ADV",
    "RBR": "ADV",
    "RBS": "ADV",
    "RP": "ADP",
    "SYM": "SYM",
    "TO": "PART",
    "UH": "INTJ",
    "VB": "VERB",
    "VBD": "VERB",
    "VBG": "VERB",
    "VBN": "VERB",
    "VBP": "VERB",
    "VBZ": "VERB",
    "WDT": "DET",
    "WP": "PRON",
    "WP$": "DET",
    "WRB": "ADV",
    "PUNCT": "PUNCT",
}
