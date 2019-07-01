
from distutils.util import strtobool
import json
import logging
import pdb
import threading
import uuid

from nnwd.models import Estimate, Predicates
from pytils import check
from pytils.log import user_log


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


DEFAULT_TOLERANCE = 0.1


class SequenceQuery:
    def __init__(self, query_engine):
        self.query_engine = query_engine

    def parse(self, data):
        predicate_strs = data["predicate"]

        if "tolerance" in data:
            tolerance = float(data["tolerance"][0])
        else:
            tolerance = DEFAULT_TOLERANCE

        return tolerance, Predicates(predicate_strs=predicate_strs)


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


class PatternMatches:
    COMMA_STANDIN = uuid.uuid4()

    def __init__(self, pattern_engine):
        self.pattern_engine = pattern_engine

    def get(self, data):
        if "tolerance" in data:
            tolerance = float(data["tolerance"][0])
        else:
            tolerance = DEFAULT_TOLERANCE

        if "skip_empties" in data:
            skip_empties = bool(strtobool(data["skip_empties"][0]))
        else:
            skip_empties = True

        if "consistent_features" in data:
            consistent_features = bool(strtobool(data["consistent_features"][0]))
        else:
            consistent_features = True

        annotated_sequences = []

        for a_s in data["annotated_sequence"]:
            a, s = a_s.split("|")
            annotation = [int(index) for index in a.split(",")]
            # Bit hacky, but should be fine.
            sequence = [("," if word == PatternMatches.COMMA_STANDIN else word) for word in s.replace(",,,", ",%s," % PatternMatches.COMMA_STANDIN).split(",")]
            annotated_sequences += [(annotation, sequence)]

        patterns = []

        for p in data["pattern"]:
            pattern = p.split(",")
            patterns += [pattern]

        logging.debug("annotated_sequences: %s" % annotated_sequences)
        logging.debug("patterns: %s" % patterns)
        return self.pattern_engine.match(tolerance, skip_empties, consistent_features, annotated_sequences, patterns)


class SoftFilters:
    def __init__(self, neural_network):
        self.neural_network = neural_network

    def get(self, data):
        sequence = data["sequence"]
        #distance = check.check_gte(int(data["distance"][0]), 0)
        return self.neural_network.soft_filters(sequence)

