
from pytils import check


class Row:
    def __init__(self, embedding, layers, softmax):
        self.embedding = embedding
        self.layers = layers
        self.softmax = softmax

    def as_json(self):
        return {
            "embedding": self.embedding.as_json(),
            "layers": [l.as_json() for l in self.layers],
            "output": self.softmax.as_json(),
        }


class Layer:
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
        self.vector = check.check_pdist(vector)
        self.colour = colour

    def as_json(self):
        return {
            "vector": [{"value": value, "position": i} for i, value in enumerate(self.vector)],
            "colour": self.colour,
        }



