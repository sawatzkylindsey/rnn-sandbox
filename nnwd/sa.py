
import logging

from nnwd import data


def create(data_dir, corpus_stream_fn):
    train_xys = []
    validation_xys = []
    test_xys = []
    sentiments = set()
    total = 0
    output_distribution = {}
    words = {}

    for triple in corpus_stream_fn():
        sentiment = get_sentiment(triple[2])
        sentiments.add(sentiment)
        xy = ([(word, None) for word in triple[1]], sentiment)

        if triple[0] == "train":
            for word in triple[1]:
                if word not in words:
                    words[word] = 0

                words[word] += 1

            train_xys += [xy]

            if sentiment not in output_distribution:
                output_distribution[sentiment] = 0

            output_distribution[sentiment] += 1
            total += 1
        elif triple[0] == "dev":
            validation_xys += [xy]
        else:
            test_xys += [xy]

    data.set_train(data_dir, train_xys)
    data.set_validation(data_dir, validation_xys)
    data.set_test(data_dir, test_xys)
    data.set_outputs(data_dir, sentiments, sentiment_sort_key)
    data.set_output_distribution(data_dir, {k: v / float(total) for k, v in output_distribution.items()})
    words = set([item[0] for item in words.items() if item[1] >= data.MINIMUM_OCCURRENCE_COUNT])
    data.set_words(data_dir, words)
    data.set_description(data_dir, data.Description(data.SA))
    logging.debug("total pairs (t, v, t): %d, %d, %d" % (sum([len(xy[0]) for xy in train_xys]), sum([len(xy[0]) for xy in validation_xys]), sum([len(xy[0]) for xy in test_xys])))
    return train_xys, validation_xys, test_xys


def get_sentiment(value):
    # [0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]
    # for very negative, negative, neutral, positive, very positive, respectively.
    if value <= 0.2:
        return "very negative"
    elif value <= 0.4:
        return "negative"
    elif value <= 0.6:
        return "neutral"
    elif value <= 0.8:
        return "positive"
    else:
        return "very positive"


def sentiment_sort_key(sentiment):
    if sentiment == "very negative":
        return 4
    elif sentiment == "negative":
        return 3
    elif sentiment == "neutral":
        return 2
    elif sentiment == "positive":
        return 1
    else:
        return 0

