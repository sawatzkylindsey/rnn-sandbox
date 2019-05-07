
import json
import logging
import pdb
import threading

from nnwd.models import Estimate
from pytils import check
from pytils.log import setup_logging, user_log


class Echo:
    def get(self, data):
        return data


class Words:
    def __init__(self, words):
        self.words = sorted([w for w in words])
        user_log.info("Vocabulary %d" % len(self.words))

    def get(self, data):
        return self.words


class Weights:
    def __init__(self, neural_network):
        self.neural_network = neural_network

    def get(self, data):
        sequence = data["sequence"]
        #distance = check.check_gte(int(data["distance"][0]), 0)
        return self.neural_network.weights(sequence)


class WeightDetail:
    def __init__(self, neural_network):
        self.neural_network = neural_network

    def get(self, data):
        sequence = data["sequence"]
        #distance = check.check_gte(int(data["distance"][0]), 0)
        part = data["part"][0]
        layer = None

        if "layer" in data:
            layer = int(data["layer"][0])

        return self.neural_network.weight_detail(sequence, part, layer)


class WeightExplain:
    def __init__(self, neural_network):
        self.neural_network = neural_network

    def get(self, data):
        sequence = data["sequence"]
        name = data["name"][0]
        column = int(data["column"][0])
        return self.neural_network.weight_explain(sequence, name, column)


class SequenceQuery:
    def __init__(self, query_engine):
        self.query_engine = query_engine

    def parse(self, data):
        predicate_strs = data["predicate"]

        if "tolerance" in data:
            tolerance = float(data["tolerance"][0])
        else:
            tolerance = 0.1

        predicates = []

        for predicate_str in predicate_strs:
            parts = {}

            for unit_targets in predicate_str.split(";"):
                unit, targets = unit_targets.split("|")
                part, layer = unit.split(",")
                features = set()

                for target in targets.split(","):
                    axis, value = target.split(":")
                    features.add((int(axis), float(value)))

                parts[(part, int(layer))] = features

            predicates += [parts]

        return tolerance, predicates


class SequenceMatchesEstimate(SequenceQuery):
    def __init__(self, *args, **kwargs):
        super(SequenceMatchesEstimate, self).__init__(*args, **kwargs)

    def get(self, data):
        tolerance, predicates = self.parse(data)

        if "exact" in data:
            rollup = self.query_engine.find(tolerance, predicates)
            return Estimate(exact=sum([sequence_match.count for sequence_match in rollup.sequence_matches]))
        else:
            return self.query_engine.find_estimate(tolerance, predicates)


class SequenceMatches(SequenceQuery):
    def __init__(self, *args, **kwargs):
        super(SequenceMatches, self).__init__(*args, **kwargs)

    def get(self, data):
        tolerance, predicates = self.parse(data)
        return self.query_engine.find(tolerance, predicates)

