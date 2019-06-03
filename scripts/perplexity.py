
import math
import sys


def make_probabilities(raw_probabilities):
    minimum = min([p for p in raw_probabilities if p > 0.0])
    transformer = lambda x: minimum / 2.0 if x == 0.0 else x
    return [transformer(p) for p in raw_probabilities]


probability_list = make_probabilities([float(p) for p in sys.argv[1:]])
total_log_probability = sum([math.log2(p) for p in probability_list])
print(2**(-(total_log_probability / len(probability_list))))

