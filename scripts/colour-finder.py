
import itertools
from nnwd import geometry


g = (127,201,127)
p = (190,174,212)
o = (253,192,134)

colour_embeddings = {
    "a": g,
    "b": p,
    "c": o,
}
interpolation_points = {
    g: ("a", .1),
    p: ("b", .1),
    o: ("c", .05)
}

#for word, probability in predictions[key].items():
#    colour = self.colour_embeddings[word]
#
#    if colour not in interpolation_points:
#        interpolation_points[colour] = (word, probability)
#    else:
#        if interpolation_points[colour][1] < probability:
#            interpolation_points[colour] = (word, probability)
#
if len(interpolation_points) == 1:
    colours[key] = "rgb(%d, %d, %d)" % next(iter(interpolation_points.keys()))
else:
    maximum_distance = None

    for pair in itertools.combinations([colour for colour in interpolation_points.keys()], 2):
        distance = geometry.distance(pair[0], pair[1])

        if maximum_distance is None or distance > maximum_distance:
            maximum_distance = distance

    lowest_probability = min([p for w, p in interpolation_points.values()])
    highest_probability = max([p for w, p in interpolation_points.values()])
    maximum_domain = highest_probability + lowest_probability
    prediction_distances = [(w, maximum_distance + (-p * maximum_distance / maximum_domain)) for w, p in interpolation_points.values()]
    fit, _ = geometry.fit_point([colour_embeddings[item[0]] for item in prediction_distances], [item[1] for item in prediction_distances], epsilon=0.1, visualize=False)
    print("rgb(%d, %d, %d)" % tuple([round(i) for i in fit]))


