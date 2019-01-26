
import logging
import math
import re
import pdb

from pytils import check


def distance(a, b):
    assert len(a) == len(b)
    return math.sqrt(sum([(a[i] - b[i])**2 for i in range(0, len(a))]))


def fit_point(reference_points, target_distances, epsilon=0.00001, visualize=False):
    assert len(reference_points) > 0
    assert len(reference_points) == len(target_distances), "%d != %d" % (len(reference_points), len(target_distances))
    assert epsilon > 0
    logging.debug("fitting: %s, %s" % (reference_points, target_distances))
    dimensions = len(reference_points[0])

    if visualize:
        assert dimensions == 2 or dimensions == 3
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        to_hex = lambda c: "#" + "".join([format(val, "02x") for val in c])

    point = [0 for i in range(0, dimensions)]

    if visualize:
        figure = plt.figure()
        axis = figure.add_subplot(111, projection="3d")
        plot_point = lambda p: list(p) + ([] if dimensions == 3 else [0])
        colouring = lambda x: to_hex([0, min((x * 20) + 100, 255), 0])

    for j, reference in enumerate(reference_points):
        if visualize:
            axis.scatter(*plot_point(reference), c=to_hex([0, 0, 255]))
            axis.text(*plot_point(reference), target_distances[j], zorder=1)

        for i in range(0, dimensions):
            point[i] += reference[i]

    count = float(len(target_distances))
    center = [p / count for p in point]

    if visualize:
        axis.scatter(*plot_point(center), c=colouring(-1))

    importance_fn = _find_importance(target_distances)
    relaxation = 1
    correction = _correction(reference_points, target_distances, center, importance_fn, relaxation)
    previous = correction
    point = [center[i] + correction[i] for i in range(0, dimensions)]
    logging.debug("center: %s -> point: %s (correction %s)" % (center, point, correction))
    t = 0

    if visualize:
        axis.scatter(*plot_point(point), c=colouring(t))

    while sum([abs(i) for i in correction]) > epsilon and t < 1000:
        t += 1
        correction = _correction(reference_points, target_distances, point, importance_fn, relaxation)

        # Make sure that each correction is bringing the point closer and closer to a solution.
        if sum([abs(i) for i in correction]) > sum([abs(i) for i in previous]):
            # If the correction starts increasing, it means we're not going to converge - so increase the relaxation step and re-loop (without apply the correction).
            relaxation += 1
            logging.debug("t=%d: r=%d, previous: %s, correction: %s" % (t, relaxation, previous, correction))
        else:
            point = [point[i] + correction[i] for i in range(0, dimensions)]
            previous = correction

            if visualize:
                axis.scatter(*plot_point(point), c=colouring(t))

        if t % 10 == 0:
            logging.debug("..fit_point.. t=%d: r=%f, %s" % (t, relaxation, point))

    if visualize:
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        #reference_points_name = re.sub("[\(\)\[\] ]", "", str(reference_points))
        #target_distances_name = re.sub("[\(\)\[\] ]", "", str(target_distances))
        #name = "%s-%s" % (reference_points_name, target_distances_name)
        #figure.savefig("internal_embedding-%s.png" % name)
        plt.draw()
        plt.pause(120)

    return point, t


def _correction(reference_points, target_distances, point, importance_fn, relaxation):
    dimensions = len(point)
    actual_distances = [distance(point, r) for r in reference_points]
    count = float(len(target_distances))
    sum_target_distances = sum(target_distances)
    correction = [0 for i in range(0, dimensions)]

    if sum_target_distances == 0:
        return correction

    sum_actual_distances = sum(actual_distances)

    if sum_actual_distances == 0:
        assert count == 1
        jostle = math.sqrt(sum_target_distances**2 / dimensions)
        return [jostle for i in range(0, dimensions)]

    for i in range(0, len(reference_points)):
        if actual_distances[i] != 0:
            delta_hypotenuse = target_distances[i] - actual_distances[i]
            importance = importance_fn(target_distances[i], relaxation)
            reference = reference_points[i]

            for dimension in range(0, dimensions):
                delta_dimension = point[dimension] - reference[dimension]
                dimension_to_hypotenuse_ratio = delta_dimension / actual_distances[i]
                c = delta_hypotenuse * dimension_to_hypotenuse_ratio
                correction[dimension] += c * importance

    return correction


def _find_importance(target_distances):
    target_maximum = max(target_distances)
    target_minimum = min(target_distances)
    delta = target_maximum - target_minimum + 1
    return lambda x, r: 1.0 / 2**((x - target_minimum + (r / 10.0)) / (delta / r**2))

