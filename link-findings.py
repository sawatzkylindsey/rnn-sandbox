
from argparse import ArgumentParser
from csv import reader as csv_reader
import os
import pdb
import sys

from nnwd import geometry
from nnwd import pickler
from pytils import adjutant


TOP = 10


def parse_sequence(sequence_str):
    assert sequence_str.startswith("[") and sequence_str.endswith("]")
    pieces = sequence_str[1:-1].split(", ")
    sequence = []

    for piece in pieces:
        if piece.startswith("'"):
            assert piece.endswith("'")
            sequence += [piece[1:-1]]
        elif piece.startswith('"'):
            assert piece.endswith('"')
            sequence += [piece[1:-1]]
        else:
            assert not piece.endswith("'")
            sequence += [piece]

    return sequence


def main(argv):
    ap = ArgumentParser(prog="server")
    ap.add_argument("findings", nargs="+")
    args = ap.parse_args(argv)
    cases = {}

    for level, finding in enumerate(args.findings):
        print("%d-%s" % (level, finding))
        with open(finding, "r") as fh:
            for row in csv_reader(fh):
                distance, prediction, expectation, backtance, word, index, sequence, point = row
                distance = float(distance)
                backtance = int(backtance)
                index = int(index)
                sequence = parse_sequence(sequence)
                case = tuple(sequence)

                if case not in cases:
                    cases[case] = {}

                if level not in cases[case]:
                    cases[case][level] = []

                cases[case][level] += [(distance, prediction, expectation, backtance, word, index)]

    matched_cases = {}
    printed = False

    for case, subd in cases.items():
        if 0 in subd:
            matches = search(0, len(args.findings) - 1, subd, None)

            if len(matches) > 0:
                best = None
                minimum = None

                for match in matches:
                    total_distance = [item[0] for item in match]

                    if minimum is None or total_distance < minimum:
                        best = match
                        minimum = total_distance

                matched_cases[case] = match

    print("matched cases: %d" % len(matched_cases))
    for case, match in sorted(matched_cases.items(), key=lambda item: item[1]):
        print(match)
        print(" ".join(case))
    return 0


def search(level, final_level, level_instances, index):
    results = []

    for instance in level_instances[level]:
        distance, prediction, expectation, instance_backtance, word, instance_index = instance

        if index is None or instance_index > index:
            if level + 1 in level_instances:
                sub_results = search(level + 1, final_level, level_instances, instance_index)

                for r in sub_results:
                    results += [[(distance, word, instance_index)] + r]
            elif level == final_level:
                # Found
                results += [[(distance, word, instance_index, prediction, expectation)]]

    return results


if __name__ == "__main__":
    main(sys.argv[1:])

