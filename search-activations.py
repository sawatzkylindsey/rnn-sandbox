
import os
import pdb

from nnwd import geometry
from nnwd import pickler
from nnwd.domain import ActivationPoint
from pytils import adjutant


RESUME_DIR = ".resume"
TOP = 10
activation_data = []


class WriteLast(Exception):
    pass


def main():
    global activation_data
    activation_data_file = os.path.join(RESUME_DIR, "activation_data.pickle")

    if os.path.exists(activation_data_file):
        activation_data = pickler.load(activation_data_file)
    else:
        raise ValueError()

    user_input = ""

    while not user_input.startswith("quit"):
        user_input = input("enter next search (part,layer|axis:target_value,..): ")

        if not user_input.startswith("quit"):
            query = None

            try:
                part, layer, query = parse(user_input)
                print("(%s, %s, %s)" % (part, layer, query))
            except WriteLast as e:
                pickler.dump((part, layer, result[:q10]), "result-q10.pickle")
                pickler.dump((part, layer, result[:q25]), "result-q25.pickle")
                pickler.dump((part, layer, result[:q50]), "result-q50.pickle")
            except Exception as e:
                print(e)
                print("error interpreting: %s" % user_input)

            if query is not None:
                result, q10, q25, q50 = find_closest(part, layer, query)
                print("found %d: " % len(result))

                for r in result[:TOP]:
                    print(r)
        else:
            # Exit path - don't do anything
            pass

    return 0


def parse(user_input):
    if user_input == "write-last":
        raise WriteLast()

    context, targets = user_input.split("|")
    contexts = context.split(",")
    part, layer = contexts
    layer = int(layer)
    query = []

    for t in targets.split(","):
        axis, target = t.split(":")
        axis = int(axis)
        target = float(target)
        query += [(axis, target)]

    return part, layer, query


def find_closest(query_part, query_layer, query):
    assert len(query) > 0, "empty query - would simply return everything!"
    global activation_data
    result = []
    minimum_distance = None
    maximum_distance = None

    for candidate in activation_data:
        if query_part == candidate.part and query_layer == candidate.layer:
            sub_point = [candidate.point[axis] for axis, _ in query]
            target_point = [target for axis, target in query]
            distance = geometry.distance(sub_point, target_point)
            result += [(distance, candidate)]

            if minimum_distance is None or distance < minimum_distance:
                minimum_distance = distance

            if maximum_distance is None or distance > maximum_distance:
                maximum_distance = distance

    if len(result) == 0:
        return result

    q50 = maximum_distance * .5
    q25 = maximum_distance * .25
    q10 = maximum_distance * .1
    print("distance stats: [%.4f, %.4f] q10: %.4f q25: %.4f q50: %.4f" % (minimum_distance, maximum_distance, q10, q25, q50))
    sorted_result = sorted(result)
    cut10 = int(len(result) * 0.1)
    cut25 = int(len(result) * 0.25)
    cut50 = int(len(result) * 0.5)

    histogram_q10, q10 = build_histogram(sorted_result[:cut10])
    histogram_q25, q25 = build_histogram(sorted_result[:cut25])
    histogram_q50, q50 = build_histogram(sorted_result[:cut50])
    print("q10: %.4f q25: %.4f q50: %.4f" % (q10, q25, q50))
    print("histograms\n  q10: %s\n  q25: %s\n  q50: %s" % (adjutant.dict_as_str(histogram_q10), adjutant.dict_as_str(histogram_q25), adjutant.dict_as_str(histogram_q50)))
    return sorted_result, cut10, cut25, cut50


def build_histogram(results):
    histogram = {}
    cutoff = None

    for r in results:
        distance, activation_point = r

        if activation_point.prediction not in histogram:
            histogram[activation_point.prediction] = 0

        histogram[activation_point.prediction] += 1
        cutoff = distance

    return histogram, cutoff


def rollup(d):
    return {k: sum([item[1] for item in v.items()]) for k, v in d.items()}


if __name__ == "__main__":
    main()

