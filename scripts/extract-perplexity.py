
from argparse import ArgumentParser
import collections
from csv import writer as csv_writer
import glob
import json
import math
import logging
import os
import pdb
import queue
import random
import re
import statistics
import sys


def main(argv):
    ap = ArgumentParser(prog="extract-perplexity")
    ap.add_argument("log_file")
    aargs = ap.parse_args(argv)
    name_series = {}
    maximum_epoch = 0

    with open(aargs.log_file, "r") as fh:
        epoch = None
        arc_epochs = None
        stored_validation = None
        stored_name = None

        for line in fh.readlines():
            name = matched_invocation(line)

            if name is not None:
                epoch = 0
                arc_epochs = 0
                stored_name = name
                name_series[stored_name] = {}

            if stored_name is not None:
                if matches_epoch(line):
                    arc_epochs += 1

                if matches_load(line):
                    arc_epochs = 0
                    stored_validation = None

                if matches_save(line):
                    epoch += arc_epochs
                    arc_epochs = 0

                    if epoch > maximum_epoch:
                        maximum_epoch = epoch

                    if stored_validation is not None:
                        name_series[stored_name][epoch] = (stored_validation, None)
                        stored_validation = None

                validation = matched_validation(line)

                if validation is not None:
                    stored_validation = validation

                test = matched_test(line)

                if test is not None:
                    name_series[stored_name][epoch] = (name_series[stored_name][epoch][0], test)

                total = matched_total(line)

                if total is not None:
                    if name_series[stored_name][epoch][1] is None:
                        name_series[stored_name][epoch] = (name_series[stored_name][epoch][0], total)

                    stored_name = None

    header = ["epoch"]

    for name, series in sorted(name_series.items()):
        header += ["%s - Dev" % name, "%s - Test" % name]

    row_columns = []

    for epoch in range(maximum_epoch):
        if any([epoch in series for name, series in name_series.items()]):
            row = [epoch]

            for name, series in sorted(name_series.items()):
                if epoch in series:
                    values = series[epoch]
                    row += ["%.4f" % values[0], "" if values[1] is None else "%.4f" % values[1]]
                else:
                    row += ["", ""]

            row_columns += [row]

    writer = csv_writer(sys.stdout)
    writer.writerow(header)

    for row in row_columns:
        writer.writerow(row)

    return 0


def matched_invocation(line):
    m = re.search("Namespace.*key_set=\[(.*)\],", line)

    if m is not None:
        key_set = json.loads("[" + m.group(1).replace("'", '"') + "]")
        return ":".join(sorted(key_set))

    return None


def matches_epoch(line):
    m = re.search("Epoch training", line)
    return m is not None


def matches_load(line):
    m = re.search("Restoring model", line)
    return m is not None


def matches_save(line):
    m = re.search("Saving model at", line)
    return m is not None


def matched_validation(line):
    m = re.search("Train arc.*-(\d+\.\d+)\)", line)

    if m is not None:
        return float(m.group(1))

    return None


def matched_test(line):
    m = re.search("Test score: -(\d+\.\d+)", line)

    if m is not None:
        return float(m.group(1))

    return None


def matched_total(line):
    m = re.search("Total perplexity: -(\d+\.\d+)", line)

    if m is not None:
        return float(m.group(1))

    return None


if __name__ == "__main__":
    ret = main(sys.argv[1:])
    sys.exit(ret)

