
import logging

from pytils import check


class Layer:
    def __init__(self, embedding, units, softmax, timestep, x_word, y_word):
        self.embedding = check.check_instance(embedding, WeightVector)
        self.units = check.check_list(units)
        self.softmax = check.check_instance(softmax, LabelWeightVector)
        self.timestep = timestep
        self.x_word = x_word
        self.y_word = y_word

    def as_json(self):
        return {
            "embedding": self.embedding.as_json(),
            "units": [u.as_json() for u in self.units],
            "softmax": self.softmax.as_json(),
            "timestep": self.timestep,
            "x_word": self.x_word,
            "y_word": self.y_word,
        }


class Unit:
    def __init__(self, remember_gate, forget_gate, output_gate, input_hat, remember, cell_previous_hat, forget, cell, cell_hat, output):
        self.remember_gate = remember_gate
        self.forget_gate = forget_gate
        self.output_gate = output_gate
        self.input_hat = input_hat
        self.remember = remember
        self.cell_previous_hat = cell_previous_hat
        self.forget = forget
        self.cell = cell
        self.cell_hat = cell_hat
        self.output = output

    def as_json(self):
        return {
            "remember_gate": self.remember_gate.as_json(),
            "forget_gate": self.forget_gate.as_json(),
            "output_gate": self.output_gate.as_json(),
            "input_hat": self.input_hat.as_json(),
            "remember": self.remember.as_json(),
            "cell_previous_hat": self.cell_previous_hat.as_json(),
            "forget": self.forget.as_json(),
            "cell": self.cell.as_json(),
            "cell_hat": self.cell_hat.as_json(),
            "output": self.output.as_json(),
        }


class WeightVector:
    def __init__(self, vector, minimum=None, maximum=None, colour="none"):
        self.vector = [float(value) for value in vector]
        self.minimum, self.maximum = canonicalize_bounds(minimum, maximum, self.vector)
        self.colour = colour

    def as_json(self):
        return {
            "vector": [{"value": value, "position": i, "column": i} for i, value in enumerate(self.vector)],
            "minimum": self.minimum,
            "maximum": self.maximum,
            "colour": self.colour,
        }


class LabelWeightVector:
    def __init__(self, label_weights, encoding, top_k=None, colour_fn=lambda word: "none"):
        self.label_weights = [(str(item[0]), float(item[1])) for item in sorted(label_weights.items(), key=lambda item: item[1], reverse=True)[:len(label_weights) if top_k is None else top_k]]
        self.encoding = encoding
        self.minimum = 0
        self.maximum = max(self.label_weights, key=lambda item: item[1])[1] * 1.25
        assert self.minimum < self.maximum, "the minimum (%s) must be less than the maximum (%s)" % (self.minimum, self.maximum)
        self.colour_fn = colour_fn

    def as_json(self):
        return {
            "vector": [{"value": item[1], "position": i, "label": item[0], "column": self.encoding[item[0]], "colour": self.colour_fn(item[0])} for i, item in enumerate(self.label_weights)],
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


def canonicalize_bounds(minimum, maximum, vector):
    if minimum is None:
        minimum = min(vector)

    if maximum is None:
        maximum = max(vector)

    if minimum > 0:
        logging.debug("pushing minimum from (%s) to (0)" % minimum)
        minimum = 0
    elif minimum >= -1 and minimum < 0:
        logging.debug("pushing minimum from (%s) to (-1)" % minimum)
        minimum = -1
    else:
        logging.debug("giving minimum 25%% leeway (%s) to (%s)" % (minimum, minimum - abs(minimum * .25)))
        minimum = minimum - abs(minimum * .25)

    if maximum < 0:
        logging.debug("pushing maximum from (%s) to (0)" % maximum)
        maximum = 0
    elif maximum >= 0 and maximum <= 1:
        logging.debug("pushing maximum from (%s) to (1)" % maximum)
        maximum = 1
    else:
        logging.debug("giving maximum 25%% leeway (%s) to (%s)" % (maximum, maximum + abs(maximum * .25)))
        maximum = maximum + abs(maximum * .25)

    assert minimum < maximum, "the minimum (%s) must be less than the maximum (%s)" % (minimum, maximum)
    return minimum, maximum
