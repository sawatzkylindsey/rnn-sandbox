
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
    ap = ArgumentParser(prog="query-data")
    ap.add_argument("-v", "--verbose", default=False, action="store_true", help="Turn on verbose logging.")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("data_dir")
    ap.add_argument("kind")
    ap.add_argument("includes", nargs="*", default=None)
    aargs = ap.parse_args(argv)
    setup_logging(".%s.log" % os.path.splitext(os.path.basename(__file__))[0], aargs.verbose, False, True, True)
    logging.debug(aargs)
    count = 0

    for xy in data.stream_data(aargs.data_dir, aargs.kind):
        # TODO: work for non-lm cases.
        sequence = [item[0] for item in xy.x] + [xy.y[-1][0]]

        if matches(sequence, aargs.includes):
            count += 1
            user_log.info(" ".join(sequence))

        if count >= aargs.limit:
            break

    return 0


def matches(sequence, includes):
    if includes is None:
        return True
    else:
        return all([include in sequence for include in includes])


if __name__ == "__main__":
    ret = main(sys.argv[1:])
    sys.exit(ret)

