
import os

from ml import nn as ffnn
from ml import base as mlbase
from pytils import adjutant
from pytils.log import setup_logging, user_log


setup_logging(".%s.log" % os.path.splitext(os.path.basename(__file__))[0], True, False, True)

KINDS = ["outputs", "cells"]
LAYERS = 2
WIDTH = 5
words = set(["abc", "def", "ghi"])
kind_labels = mlbase.Labels(set(KINDS))
layer_labels = mlbase.Labels(set(range(LAYERS)))
activation_vector = mlbase.VectorField(WIDTH)
predictor_input = mlbase.ConcatField([kind_labels, layer_labels, activation_vector])
predictor_output = mlbase.Labels(words)
predictor = ffnn.Model("predictor", ffnn.HyperParameters().width(10).layers(1), predictor_input, predictor_output, mlbase.SINGLE_LABEL)

data = [
    mlbase.Xy(("outputs", 0, [.1, .2, .3, .4, .5]), {"abc": .6, "def": .2, "ghi": .2}),
    mlbase.Xy(("outputs", 1, [.1, .2, .3, .4, .5]), {"abc": .1, "def": .6, "ghi": .3}),
    #mlbase.Xy(("outputs", 1, [.1, .2, .3, .4, .5]), {"abc": .3, "def": .3, "ghi": .4}),
    #mlbase.Xy(("outputs", 1, [.5, .4, .3, .2, .1]), {"abc": .4, "def": .4, "ghi": .2}),
    #mlbase.Xy(("cells", 0, [.1, .2, .3, .4, .5]), {"abc": .2, "def": .4, "ghi": .4}),
    #mlbase.Xy(("cells", 0, [.5, .4, .3, .2, .1]), {"abc": .6, "def": .2, "ghi": .2}),
    #mlbase.Xy(("cells", 1, [.1, .2, .3, .4, .5]), {"abc": .35, "def": .35, "ghi": .3}),
    #mlbase.Xy(("cells", 1, [.5, .4, .3, .2, .1]), {"abc": .3, "def": .3, "ghi": .4}),
]

loss = predictor.train(data, mlbase.TrainingParameters().epochs(2000).relative(0.000005).absolute(0.5).batch(1))
print("loss: %s" % loss)

for xy in data:
    inferred = predictor.evaluate(xy.x).distribution
    print("%s:\n  expected: %s\n    actual: %s" % (xy.x, adjutant.dict_as_str(xy.y, use_key=False), adjutant.dict_as_str(inferred, use_key=False)))

print("inference")
test_data = [
    ("outputs", 0, [.1, .2, .3, .4, .5]),
    ("outputs", 0, [.2, .2, .3, .4, .5]),
    ("outputs", 0, [.2, .2, .3, .4, .5]),
    #("cells", 1, [.1, .2, .3, .3, .5]),
    #("cells", 1, [.1, .2, .2, .3, .5]),
    #("cells", 1, [.2, .2, .2, .3, .5]),
]

#for x in test_data:
#    inferred = predictor.evaluate(x).distribution
#    print("%s:\n  predicted: %s" % (x, adjutant.dict_as_str(inferred, use_key=False)))
inferred = predictor.evaluate(test_data)
for q, i in enumerate(inferred):
    print("%s:\n  predicted: %s" % (q, adjutant.dict_as_str(i.distribution, use_key=False)))

