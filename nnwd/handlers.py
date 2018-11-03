
import logging
import pdb
import threading

from pytils.log import setup_logging, user_log
from nnwd.models import Layer, Unit, WeightVector, LabelWeightVector
from nnwd import rnn


class Echo:
    def get(self, data):
        return data


class NeuralNetwork:
    def __init__(self, words, xy_sequences):
        self.neural_network = rnn.Rnn(1, 5, words)
        self._background_training = threading.Thread(target=self.neural_network.train, args=([[rnn.Xy(t[0], t[1]) for t in sequence] for sequence in xy_sequences], 100, True))
        self._background_training.daemon = True
        self._background_training.start()

    def get(self, data):
        self._background_training.join()
        stepwise_rnn = self.neural_network.stepwise()
        result, instruments = stepwise_rnn.step("the", ["embedding", "units"])
        embedding = WeightVector(instruments["embedding"])
        units = [self._unit(u) for u in instruments["units"]]
        softmax = LabelWeightVector(result.distribution)
        return Layer(embedding, units, softmax)

    def _unit(self, unit):
        return Unit(
            WeightVector(unit),
            WeightVector([0,0,0,0,0], -1, 1),
            WeightVector(unit),
            WeightVector(unit),
            WeightVector(unit),
            WeightVector(unit),
            WeightVector(unit),
            WeightVector(unit),
        )

