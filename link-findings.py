
from argparse import ArgumentParser
import collections
import os
import pdb
import sys

from nnwd import geometry
from nnwd import pickler
from nnwd.domain import ActivationPoint
from pytils import adjutant


TOP = 10
MatchPoint = collections.namedtuple("MatchPoint", ["distance", "word", "index", "prediction", "expectation"])


def main(argv):
    ap = ArgumentParser(prog="server")
    ap.add_argument("-e", "--elastic", action="store_true", default=False)
    ap.add_argument("findings", nargs="+")
    args = ap.parse_args(argv)
    cases = {}
    level_part_layers = []

    for level_finding in args.findings:
        level, finding = level_finding.split(":")
        level = int(level)
        print("%d:%s" % (level, finding))
        part, layer, data = pickler.load(finding)
        assert (level, part, layer) not in level_part_layers, "duplicate (level, part, layer): (%s, %s, %s)" % (level, part, layer)
        level_part_layers += [(level, part, layer)]

        for i in data:
            distance, activation_point = i
            case = tuple(activation_point.sequence)

            if case not in cases:
                cases[case] = {}

            if (level, part, layer) not in cases[case]:
                cases[case][(level, part, layer)] = []

            cases[case][(level, part, layer)] += [i]

    level_part_layers = sorted(level_part_layers)
    print(level_part_layers)
    matched_cases = {}
    printed = False

    for case, subd in cases.items():
        matches = search(0, level_part_layers, subd, None)

        if len(matches) > 0:
            best = None
            minimum = None

            for match in matches:
                total_distance = sum([item[0] for item in match])

                if minimum is None or total_distance < minimum:
                    best = match
                    minimum = total_distance

            matched_cases[case] = (total_distance, match)

    print("matched cases: %d" % len(matched_cases))
    for case, match in sorted(matched_cases.items(), key=lambda item: item[1]):
        print(match)
        print(" ".join(case))
    return 0


def search(constraint_index, level_part_layers, lpl_instances, index):
    results = []
    constraint = level_part_layers[constraint_index]

    # This case doesn't even have instances across all the constraining (level, part, layers) - definitely can't be satisified.
    if len(lpl_instances) < len(level_part_layers):
        return []

    if constraint_index > 0:
        previous_level = level_part_layers[constraint_index - 1][0]
        assert previous_level <= constraint[0]

        if previous_level == constraint[0]:
            acceptable = lambda ap: ap.index == index
        else:
            acceptable = lambda ap: ap.index > index
    else:
        acceptable = lambda ap: True

    for instance in lpl_instances[constraint]:
        distance, activation_point = instance

        if index is None or acceptable(activation_point):
            word = activation_point.sequence[activation_point.index]

            # If we're at the final constraint.
            if constraint_index + 1 == len(level_part_layers):
                # Found
                results += [[MatchPoint(distance=distance, word=word, index=activation_point.index, prediction=activation_point.prediction, expectation=activation_point.expectation)]]
            else:
                sub_results = search(constraint_index + 1, level_part_layers, lpl_instances, activation_point.index)

                for r in sub_results:
                    results += [[MatchPoint(distance=distance, word=word, index=activation_point.index, prediction=None, expectation=None)] + r]

    return results


if __name__ == "__main__":
    main(sys.argv[1:])

