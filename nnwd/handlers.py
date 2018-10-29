
import logging
import pdb

from pytils.log import setup_logging, user_log
from nnwd.models import Layer, Weight


class Echo:
    def get(self, data):
        return data


class NeuralNetwork:
    def get(self, data):
        return Layer(Weight([.25, .25, .5]), Weight([.1, .8, .1]), Weight([.5, .25, .25]))


