
from pytils import check


class Layer:
    def __init__(self, embedding, units, softmax):
        self.embedding = check.check_instance(embedding, Weight)
        self.units = check.check_list(units)
        self.softmax = check.check_instance(softmax, LabelWeight)

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


class Weight:
    def __init__(self, vector, colour="#000000"):
        self.vector = check.check_iterable(vector)
        self.colour = colour

    def as_json(self):
        return {
            "vector": [{"value": float(value), "position": i} for i, value in enumerate(self.vector)],
            "colour": self.colour,
        }


class LabelWeight:
    def __init__(self, label_weights):
        self.label_weights = [item for item in sorted(label_weights.items(), key=lambda item: item[1], reverse=True)]

    def as_json(self):
        return {
            "vector": [{"value": float(item[1]), "position": i, "label": str(item[0])} for i, item in enumerate(self.label_weights)],
        }

