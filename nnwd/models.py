
import logging
import pdb

from pytils import check


class Timestep:
    def __init__(self, embedding, units, softmax, timestep, x_word, y_word):
        self.embedding = check.check_instance(embedding, HiddenState)
        self.units = check.check_dict(units)
        self.softmax = check.check_instance(softmax, LabelDistribution)
        self.timestep = timestep
        self.x_word = x_word
        self.y_word = y_word

    def as_json(self):
        return {
            "embedding": self.embedding.as_json(),
            "units": {k: {l: v.as_json() for l, v in subd.items()} for k, subd in self.units.items()},
            "softmax": self.softmax.as_json(),
            "timestep": self.timestep,
            "x_word": self.x_word,
            "y_word": self.y_word,
        }


class Unit:
    def __init__(self, remember_gate, forget_gate, output_gate, input_hat, remember, cell_previous, forget, cell, cell_hat, output):
        self.remember_gates = remember_gate
        self.forget_gates = forget_gate
        self.output_gates = output_gate
        self.input_hats = input_hat
        self.remembers = remember
        self.cell_previouses = cell_previous
        self.forgets = forget
        self.cells = cell
        self.cell_hats = cell_hat
        self.outputs = output

    def as_json(self):
        return {
            "remember_gates": self.remember_gates.as_json(),
            "forget_gates": self.forget_gates.as_json(),
            "output_gates": self.output_gates.as_json(),
            "input_hats": self.input_hats.as_json(),
            "remembers": self.remembers.as_json(),
            "cell_previouses": self.cell_previouses.as_json(),
            "forgets": self.forgets.as_json(),
            "cells": self.cells.as_json(),
            "cell_hats": self.cell_hats.as_json(),
            "outputs": self.outputs.as_json(),
        }


class HiddenState:
    def __init__(self, name, name_no_t, vector, min_max=(None, None), colour=None, predictions=None, positioning=None):
        self.name = name
        self.name_no_t = name_no_t
        self.vector = [float(value) for value in vector]
        self.minimum, self.maximum = canonicalize_bounds(min_max, self.vector)
        self.colour = colour
        self.predictions = None if predictions is None else check.check_instance(predictions, LabelDistribution)
        self.positioning = positioning

    def as_json(self):
        return {
            "name": self.name,
            "name_no_t": self.name_no_t,
            "vector": [{"value": value, "position": i, "actual_position": None if self.positioning is None else self.positioning[i]} for i, value in enumerate(self.vector)],
            "minimum": self.minimum,
            "maximum": self.maximum,
            "colour": self.colour,
            "predictions": None if self.predictions is None else self.predictions.as_json(),
        }


class LabelDistribution:
    def __init__(self, name, label_weights, sort_key=lambda key_value: -key_value[1], top_k=None, colour_fn=lambda i: None):
        self.name = name
        self.label_weights = [(str(item[0]), float(item[1])) for item in sorted(label_weights.items(), key=sort_key)[:len(label_weights) if top_k is None else top_k]]
        self.minimum = 0
        self.maximum = max(self.label_weights, key=lambda item: item[1])[1] * 1.25
        assert self.minimum < self.maximum, "the minimum (%s) must be less than the maximum (%s)" % (self.minimum, self.maximum)
        self.colour_fn = colour_fn

    def as_json(self):
        return {
            "name": self.name,
            "vector": [{"value": item[1], "position": i, "label": item[0], "colour": self.colour_fn(item[0])} for i, item in enumerate(self.label_weights)],
            "minimum": self.minimum,
            "maximum": self.maximum
        }


class WeightExplain:
    def __init__(self, vectors, bias, bound=None):
        self.vectors = {name: [float(value) for value in vector] for name, vector in vectors.items()}
        self.bias = float(bias)
        self.bound = bound

        if bound is None:
            minimum = min([value for vector in self.vectors.values() for value in vector])
            maximum = max([value for vector in self.vectors.values() for value in vector])
            self.bound = max(abs(minimum), abs(maximum))

    def as_json(self):
        return {
            "vectors": {name: [{"value": value, "position": i} for i, value in enumerate(vector)] for name, vector in self.vectors.items()},
            "bound": self.bound,
            "bias": self.bias,
        }


class WeightDetail:
    def __init__(self, mini, full, back_links):
        self.mini = mini
        self.full = full
        self.back_links = back_links

    def as_json(self):
        return {
            "mini": self.mini.as_json(),
            "full": self.full.as_json(),
            "back_links": self.back_links
        }


class SequenceRollup:
    def __init__(self, sequence_matches):
        self.sequence_matches = sequence_matches

    def as_json(self):
        return {
            "sequence_matches": [sm.as_json() for sm in self.sequence_matches]
        }


class SequenceMatch:
    def __init__(self, words, elides, count):
        assert len(elides) == len(words) + 1
        self.words = words
        self.elides = elides
        self.count = check.check_gte(count, 1)

    def as_json(self):
        return {
            "words": self.words,
            "elides": self.elides,
            "count": self.count
        }


class Estimate:
    def __init__(self, lower=None, upper=None, exact=None):
        self.lower = lower
        self.upper = upper
        self.exact = exact

    def as_json(self):
        return {
            "lower": self.lower,
            "upper": self.upper,
            "exact": self.exact,
        }


class SoftFilters:
    def __init__(self, words, timestep_units):
        self.words = words
        timesteps = len(timestep_units)
        layers = len(timestep_units[0][0])
        self.matrix_units = [[[None] * layers for column in range(timesteps)] for row in range(timesteps)]

        # Square out the data into a matrix.
        for row in range(timesteps):
            for column in range(timesteps):
                if column < len(timestep_units[row]):
                    for layer in range(layers):
                        self.matrix_units[row][column][layer] = timestep_units[row][column][layer]


def canonicalize_bounds(min_max, vector):
    if min_max[0] is None:
        minimum = min(vector)
    else:
        minimum = min_max[0]

    if min_max[1] is None:
        maximum = max(vector)
    else:
        maximum = min_max[1]

    if minimum > 0:
        minimum = 0
    elif minimum >= -1 and minimum < 0:
        minimum = -1
    else:
        minimum = minimum - abs(minimum * .25)

    if maximum < 0:
        maximum = 0
    elif maximum >= 0 and maximum <= 1:
        maximum = 1
    else:
        maximum = maximum + abs(maximum * .25)

    assert minimum < maximum, "the minimum (%s) must be less than the maximum (%s)" % (minimum, maximum)
    return minimum, maximum


class Predicates:
    def __init__(self, predicates=None, predicate_strs=None):
        assert predicates is not None or predicate_strs is not None

        if predicates is not None:
            assert predicate_strs is None
            self.predicates = predicates
            mirror = Predicates(predicate_strs=self.as_strs())
            assert self == mirror, "%s != %s" % (str(self), str(mirror))

        if predicate_strs is not None:
            assert predicates is None

            def typify(feature):
                axis, value = feature.split(":")
                return int(axis), float(value)

            self.predicates = []

            for predicate_str in predicate_strs:
                constraints = {}

                for key_features in predicate_str.split(";"):
                    key, features = key_features.split("|")
                    constraints[key] = [typify(feature) for feature in features.split(",")]

                self.predicates += [constraints]

    def as_strs(self):
        return [";".join(["%s|%s" % (key, ",".join(["%s:%s" % (axis, value) for axis, value in item])) for key, item in predicate.items()]) for predicate in self.predicates]

    def as_json(self):
        return {
            "predicates": self.predicates,
            "shortcut": "&predicate=".join(self.as_strs()),
        }

    def levels(self):
        return [(level, predicate) for level, predicate in enumerate(self.predicates)]

    def __eq__(self, other):
        if other is None:
            return False

        return self.predicates == other.predicates

    def __repr__(self):
        return "Predicates{levels=%d, keys=%d}" % (len(self.predicates), sum([len(predicate) for predicate in self.predicates]))

