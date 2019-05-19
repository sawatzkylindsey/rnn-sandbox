
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
import threading

from ml import base as mlbase
from ml import model
from ml import scoring
from nnwd import data
from nnwd import parameters
from nnwd import pickler
from nnwd import query
from nnwd import reduction
from nnwd import rnn
from nnwd import semantic
from nnwd import sequential
from nnwd import states

from pytils import adjutant
from pytils.log import setup_logging, teardown, user_log


BATCH_SIZE = 100


@teardown
def main(argv):
    ap = ArgumentParser(prog="generate-query-database")
    ap.add_argument("-v", "--verbose", default=False, action="store_true", help="Turn on verbose logging.")
    ap.add_argument("data_dir")
    ap.add_argument("sequential_dir")
    ap.add_argument("activation_dir")
    ap.add_argument("query_dir")
    aargs = ap.parse_args(argv)
    setup_logging(".%s.log" % os.path.splitext(os.path.basename(__file__))[0], aargs.verbose, False, True, True)
    logging.debug(aargs)

    lstm = sequential.load_model(aargs.data_dir, aargs.sequential_dir)
    threads = []

    for key in lstm.keys():
        thread = threading.Thread(target=generate_db, args=[lstm, aargs.activation_dir, key, aargs.query_dir])
        # Non-daemon threads will keep the program running until they finish (as per documentation).
        thread.daemon = False
        thread.start()
        threads += [thread]

    for thread in threads:
        thread.join()

    return 0


def generate_db(lstm, activation_dir, key, query_dir):
    logging.debug("Processing activation data for query database %s." % key)
    query_db = query.database_for(query_dir, lstm, key)
    sequence_ids = {}
    batch = []

    for i, sequence_index_point in enumerate(states.stream_activations(activation_dir, key)):
        if i % 100000 == 0:
            logging.debug("At the %d-hundred-Kth instance of %s." % (int(i / 100000), key))
            query_db.commit()

        sequence = tuple([word_pos[0] for word_pos in sequence_index_point[0]])
        sequence_index = sequence_index_point[1]
        point = sequence_index_point[2]
        #point = tuple([float(v) for v in sequence_index_point[2]])

        if sequence not in sequence_ids:
            sequence_id = query_db.insert_sequence(sequence)
            sequence_ids[sequence] = sequence_id
        else:
            sequence_id = sequence_ids[sequence]

        data = (sequence_id, sequence_index) + point
        batch += [data]

        if len(batch) == BATCH_SIZE:
            query_db.insert_activations(batch)
            batch = []

    if len(batch) > 0:
        query_db.insert_activations(batch)

    query_db.commit()
    logging.debug("Finished activation data for query database %s." % key)


if __name__ == "__main__":
    ret = main(sys.argv[1:])
    sys.exit(ret)

