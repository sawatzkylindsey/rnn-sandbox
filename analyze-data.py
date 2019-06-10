
from argparse import ArgumentParser
import collections
from csv import writer as csv_writer
import glob
import math
import logging
import os
import pdb
import queue
import random
import statistics
import sys

from nnwd import data
from nnwd import sequential
from nnwd import states
from nnwd import parameters
from nnwd import pickler
from nnwd import rnn

from pytils import adjutant
from pytils.log import setup_logging, teardown, user_log


@teardown
def main(argv):
    ap = ArgumentParser(prog="analyze-data")
    ap.add_argument("-v", "--verbose", default=False, action="store_true", help="Turn on verbose logging.")
    ap.add_argument("data_dir")
    aargs = ap.parse_args(argv)
    setup_logging(".%s.log" % os.path.splitext(os.path.basename(__file__))[0], aargs.verbose, False, True, True)
    logging.debug(aargs)

    analyze(data.stream_train(aargs.data_dir), "train")
    analyze(data.stream_test(aargs.data_dir), "test")
    return 0


def analyze(stream, kind):
    count = 0
    length = 0

    for item in stream:
        count += 1
        length += len(item.x)

    user_log.info("%s average length: %.4f" % (kind, length / float(count)))


if __name__ == "__main__":
    ret = main(sys.argv[1:])
    sys.exit(ret)

