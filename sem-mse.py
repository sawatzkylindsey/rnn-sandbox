
from argparse import ArgumentParser
import collections
from csv import writer as csv_writer
import os
import pdb
import sys

from nnwd import pickler


DISTRIBUTIONS = "sem-distributions.pickle"
OUTPUT = "output-distribution.pickle"


# Semantic encoding model - mean squared error
def main(argv):
    ap = ArgumentParser(prog="sem-mse")
    ap.add_argument("resume_a")
    ap.add_argument("resume_b")
    args = ap.parse_args(argv)

    output_distributions = [i for i in pickler.load(os.path.join(args.resume_a, OUTPUT))]
    assert len(output_distributions) == 1
    output_distribution = output_distributions[0]
    size = None
    uniform_distribution = None
    stream_a = pickler.load(os.path.join(args.resume_a, DISTRIBUTIONS))
    stream_b = pickler.load(os.path.join(args.resume_b, DISTRIBUTIONS))
    count = 0
    comparison_total = 0.0
    uniform_total_a = 0.0
    uniform_total_b = 0.0
    distribution_total_a = 0.0
    distribution_total_b = 0.0

    for distribution_a, distribution_b in zip(stream_a, stream_b):
        assert len(distribution_a) == len(distribution_b)

        if size is None:
            size = len(distribution_a)
            value = 1.0 / size
            uniform_distribution = {value for key in distribution_a.keys()}

        comparison_total += sum_squared_error(distribution_a, distribution_b)
        uniform_total_a += sum_squared_error(distribution_a, uniform_distribution)
        uniform_total_b += sum_squared_error(distribution_b, uniform_distribution)
        distribution_total_a += sum_squared_error(distribution_a, output_distribution)
        distribution_total_b += sum_squared_error(distribution_b, output_distribution)
        count += 1

    with open("sem-mse-analysis.csv", "w") as fh:
        writer = csv_writer(fh)
        writer.writerow(["comparison", "sum of squared error", "mean squared error", "mse normalized"])
        writer.writerow(row_data("comparison", comparison_total, count, size))
        writer.writerow(row_data("uniform a", uniform_total_a, count, size))
        writer.writerow(row_data("uniform b", uniform_total_b, count, size))
        writer.writerow(row_data("distribution a", distribution_total_a, count, size))
        writer.writerow(row_data("distribution b", distribution_total_b, count, size))

    return 0


def row_data(name, total, count, dimensions):
    return [name, total, (total / count), (total / (count * dimensions))]


def sum_squared_error(main, comparable):
    total = 0.0

    for key in main.keys():
        value = comparable[key] if key in comparable else 0.0
        total += (main[key] - value)**2

    return total


if __name__ == "__main__":
    main(sys.argv[1:])

