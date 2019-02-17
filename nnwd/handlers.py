
import json
import logging
import pdb
import threading

from pytils import check
from pytils.log import setup_logging, user_log


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
        distance = check.check_gte(int(data["distance"][0]), 0)
        return self.neural_network.weights(sequence, distance)


class WeightDetail:
    def __init__(self, neural_network):
        self.neural_network = neural_network

    def get(self, data):
        sequence = data["sequence"]
        distance = check.check_gte(int(data["distance"][0]), 0)
        part = data["part"][0]
        layer = None

        if "layer" in data:
            layer = int(data["layer"][0])

        return self.neural_network.weight_detail(sequence, distance, part, layer)


class WeightExplain:
    def __init__(self, neural_network):
        self.neural_network = neural_network

    def get(self, data):
        sequence = data["sequence"]
        name = data["name"][0]
        column = int(data["column"][0])
        return self.neural_network.weight_explain(sequence, name, column)


