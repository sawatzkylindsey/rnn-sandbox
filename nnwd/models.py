
import logging

from pytils import check


class Layer:
    def __init__(self, embedding, units, softmax, timestep):
        self.embedding = check.check_instance(embedding, WeightVector)
        self.units = (units)
        self.softmax = check.check_instance(softmax, LabelWeightVector)
        self.timestep = timestep

    def as_json(self):
        return {
            "embedding": self.embedding.as_json(),
            "units": [u.as_json() for u in self.units],
            "softmax": self.softmax.as_json(),
            "timestep": self.timestep,
        }


class Unit:
    def __init__(self, remember_gate, forget_gate, output_gate, input_hat, remember, cell_previous, forget, cell, cell_hat, output):
        self.remember_gate = remember_gate
        self.forget_gate = forget_gate
        self.output_gate = output_gate
        self.input_hat = input_hat
        self.remember = remember
        self.cell_previous = cell_previous
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
            "cell_previous": self.cell_previous.as_json(),
            "forget": self.forget.as_json(),
            "cell": self.cell.as_json(),
            "cell_hat": self.cell_hat.as_json(),
            "output": self.output.as_json(),
        }


class WeightVector:
    def __init__(self, vector, minimum=None, maximum=None, colour="none"):
        self.vector = [float(value) for value in vector]
        self.minimum = minimum if minimum is not None else min(self.vector)
        self.maximum = maximum if maximum is not None else max(self.vector)

        if self.minimum >= 0 and self.minimum <= 1:
            logging.debug("pushing minimum from (%s) to (0)" % self.minimum)
            self.minimum = 0
        elif self.minimum >= -1 and self.minimum <= 0:
            logging.debug("pushing minimum from (%s) to (-1)" % self.minimum)
            self.minimum = -1
        else:
            logging.debug("giving minimum 25%% leeway (%s) to (%s)" % (self.minimum, self.minimum - abs(self.minimum * .25)))
            self.minimum = self.minimum - abs(self.minimum * .25)

        if self.maximum >= 0 and self.maximum <= 1:
            logging.debug("pushing maximum from (%s) to (1)" % self.maximum)
            self.maximum = 1
        else:
            logging.debug("giving maximum 25%% leeway (%s) to (%s)" % (self.maximum, self.maximum + abs(self.maximum * .25)))
            self.maximum = self.maximum + abs(self.maximum * .25)

        assert self.minimum < self.maximum, "the minimum (%s) must be less than the maximum (%s)" % (self.minimum, self.maximum)
        self.colour = colour

    def as_json(self):
        return {
            "vector": [{"value": value, "position": i} for i, value in enumerate(self.vector)],
            "minimum": self.minimum,
            "maximum": self.maximum,
            "colour": self.colour,
        }


class LabelWeightVector:
    def __init__(self, label_weights):
        self.label_weights = [(str(item[0]), float(item[1])) for item in sorted(label_weights.items(), key=lambda item: item[1], reverse=True)]
        self.minimum = 0
        self.maximum = max(self.label_weights, key=lambda item: item[1])[1] * 1.25
        assert self.minimum < self.maximum, "the minimum (%s) must be less than the maximum (%s)" % (self.minimum, self.maximum)

    def as_json(self):
        return {
            "vector": [{"value": item[1], "position": i, "label": item[0]} for i, item in enumerate(self.label_weights)],
            "minimum": self.minimum,
            "maximum": self.maximum
        }

