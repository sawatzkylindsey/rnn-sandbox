
from pytils import check


class Layer:
    def __init__(self, embedding, hidden, output):
        self.embedding = embedding
        self.hidden = hidden
        self.output = output

    def as_dict(self):
        return {
            "embedding": self.embedding.as_dict(),
            "hidden": self.hidden.as_dict(),
            "output": self.output.as_dict(),
        }

class Weight:
    def __init__(self, vector, colour="#000000"):
        self.vector = check.check_pdist(vector)
        self.colour = colour

    def as_dict(self):
        return {
            "vector": [{"value": value, "position": i} for i, value in enumerate(self.vector)],
            "colour": self.colour,
        }



