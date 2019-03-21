
import hashlib
import logging
import math
import re
import pdb

from pytils import check


def distance(a, b):
    return hypotenuse(deltas(a, b))


def deltas(a, b):
    assert len(a) == len(b)
    return [abs(a[i] - b[i]) for i in range(0, len(a))]


def hypotenuse(vector):
    return math.sqrt(sum([part**2 for part in vector]))


def fit_proportion(reference_points, target_proportions):
    assert len(reference_points) == 2
    assert len(target_proportions) == 2
    dimensions = len(reference_points[0])
    total_distance = distance(*reference_points)
    distance_a = total_distance * target_proportions[0]
    distance_b = total_distance * target_proportions[1]
    correction = _correction(reference_points, [distance_a, distance_b], reference_points[0], lambda x, r: 1, None)
    return [reference_points[0][i] + correction[i] for i in range(0, dimensions)]


def fit_point(reference_points, target_distances, epsilon=0.00001, visualize=False):
    assert len(reference_points) > 0
    assert len(reference_points) == len(target_distances), "%d != %d" % (len(reference_points), len(target_distances))
    assert epsilon > 0
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
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        plot_point = lambda p: list(p) + ([] if dimensions == 3 else [0])
        colouring = lambda x: to_hex([0, min((x * 5) + 100, 255), 0])

    for j, reference in enumerate(reference_points):
        if visualize:
            axis.scatter(*plot_point(reference), c=to_hex([0, 0, 255]))
            axis.text(*plot_point(reference), "%.2f" % target_distances[j], zorder=1)

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
        else:
            point = [point[i] + correction[i] for i in range(0, dimensions)]
            previous = correction

            if visualize:
                axis.scatter(*plot_point(point), c=colouring(t))

        if t % 10 == 0:
            pass

    if visualize:
        axis.scatter(*plot_point(point), c=colouring(t))
        axis.text(*plot_point(point), "%.2f, %.2f, %.2f" % tuple(plot_point(point)), zorder=1)
        #m = hashlib.sha256()
        #m.update(re.sub("[\(\)\[\] ]", "", str(reference_points)).encode("utf-8"))
        #m.update(re.sub("[\(\)\[\] ]", "", str(target_distances)).encode("utf-8"))
        #figure.savefig("internal_embedding-%s.png" % m.hexdigest())
        #plt.draw()
        #plt.pause(120)

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
    #
    # Base form:
    #     1
    # __________
    #   x/delta
    #  2
    #
    # The idea is use this base form, but push the curve so that the mininum input results in ~1 and the maximum input results in ~.7.
    # This means we'll adhere to the change required to the minimum inputs ~100%, and the change required to the maximum inputs ~70%.
    #
    # Never output exactly 1, which would overfocus on the minimum input.
    # This also needs to scale with relaxation, otherwise the far points eventually degrade while the close point remain ~100% salient.
    #                                                    vvvvvvvvvvvvvvvvvvvvvvvvvv
    # This is what pushes the maximum input to ~.7.  Setting the 2.0 to 1.0 will push to precisely .5.
    #                                                                                    vvvvvvvvvvvvvvvvvvvv
    return lambda x, r: 1.0 / 2**((x - target_minimum + (0.1 * (target_minimum + r))) / (delta * (2.0 / r**2)))

