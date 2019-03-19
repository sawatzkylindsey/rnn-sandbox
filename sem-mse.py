
from argparse import ArgumentParser
import collections
import os
import pdb
import sys

from nnwd import pickler


DISTRIBUTIONS = "sem-distributions.pickle"


# Semantic encoding model - mean squared error
def main(argv):
    ap = ArgumentParser(prog="sem-mse")
    ap.add_argument("resume_a")
    ap.add_argument("resume_b")
    args = ap.parse_args(argv)

    stream_a = pickler.load(os.path.join(args.resume_a, DISTRIBUTIONS))
    stream_b = pickler.load(os.path.join(args.resume_b, DISTRIBUTIONS))
    count = 0
    dimensions = None
    total = 0.0

    for distribution_a, distribution_b in zip(stream_a, stream_b):
        value = ssel(distribution_a, distribution_b)
        count += 1
        total += value

        if dimensions is None:
            dimensions = len(distribution_a)

    print("sum of squared error: %s" % total)
    print("  mean squared error: %s" % (total / count))
    print("      mse normalized: %s" % (total / (count * dimensions)))
    return 0


def ssel(a, b):
    total = 0.0

    for key in a.keys():
        total += (a[key] - b[key])**2

    return total


if __name__ == "__main__":
    main(sys.argv[1:])

