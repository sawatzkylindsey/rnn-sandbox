
import logging
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
    return train_xys, validation_xys, test_xys

