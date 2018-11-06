
from pytils import check


class Layer:
    def __init__(self, embedding, units, softmax):
        self.embedding = check.check_instance(embedding, WeightVector)
        self.units = check.check_list(units)
        self.softmax = check.check_instance(softmax, LabelWeightVector)

    def as_json(self):
        return {
            "embedding": self.embedding.as_json(),
            "units": [u.as_json() for u in self.units],
            "softmax": self.softmax.as_json(),
        }


class Unit:
    def __init__(self, forget_gate, remember_gate, output_gate, c_previous, c_hat, c, input_hat, output):
        self.forget_gate = forget_gate
        self.remember_gate = remember_gate
        self.output_gate = output_gate
        self.c_previous = c_previous
        self.c_hat = c_hat
        self.c = c
        self.input_hat = input_hat
        self.output = output

    def as_json(self):
        return {
            "forget_gate": self.forget_gate.as_json(),
            "remember_gate": self.remember_gate.as_json(),
            "output_gate": self.output_gate.as_json(),
            "c_previous": self.c_previous.as_json(),
            "c_hat": self.c_hat.as_json(),
            "c": self.c.as_json(),
            "input_hat": self.input_hat.as_json(),
            "output": self.output.as_json(),
        }


class WeightVector:
    def __init__(self, vector, minimum=None, maximum=None, colour="none"):
        self.vector = [float(value) for value in vector]
        self.minimum = minimum if minimum is not None else min(self.vector)
        self.maximum = maximum if maximum is not None else max(self.vector)
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
        self.maximum = max(self.label_weights, key=lambda item: item[1])[1] * 1.5

    def as_json(self):
        return {
            "vector": [{"value": item[1], "position": i, "label": item[0]} for i, item in enumerate(self.label_weights)],
            "minimum": 0,
            "maximum": self.maximum
        }

