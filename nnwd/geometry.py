
import math
import pdb

from pytils import check


def distance(a, b):
    check.check_list(a)
    check.check_list(b)
    assert len(a) == len(b)
    return math.sqrt(sum([(a[i] - b[i])**2 for i in range(0, len(a))]))


def fit_point(reference_points, target_distances, epsilon=0.00001, visualize=False):
    check.check_list(reference_points)
    check.check_list(target_distances)
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
        plot_point = lambda p: p + ([] if dimensions == 3 else [0])
        colouring = lambda x: to_hex([0, min((x * 10) + 100, 255), 0])

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

    t = 0
    correction = _correction(reference_points, target_distances, center)
    point = [center[i] + correction[i] for i in range(0, dimensions)]

    if visualize:
        axis.scatter(*plot_point(point), c=colouring(t))

    while sum([abs(i) for i in correction]) > epsilon:
        t += 1
        correction = _correction(reference_points, target_distances, point)
        point = [point[i] + correction[i] for i in range(0, dimensions)]

        if visualize:
            axis.scatter(*plot_point(point), c=colouring(t))

    if visualize:
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        plt.draw()
        plt.pause(60)

    return point, t


def _correction(reference_points, target_distances, point):
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

    # Close distances should be respected more than far ones.
    importance_fn = lambda x: (1.0 / (1.2**x)) - 0.2

    for i in range(0, len(reference_points)):
        if actual_distances[i] != 0:
            delta_hypotenuse = target_distances[i] - actual_distances[i]
            importance = importance_fn(target_distances[i])
            reference = reference_points[i]

            for dimension in range(0, dimensions):
                delta_dimension = point[dimension] - reference[dimension]
                dimension_to_hypotenuse_ratio = delta_dimension / actual_distances[i]
                c = delta_hypotenuse * dimension_to_hypotenuse_ratio
                correction[dimension] += c * importance

    return correction

