
from csv import writer as csv_writer
import os
import pdb

from nnwd import geometry
from nnwd import pickler
from pytils import adjutant


RESUME_DIR = ".resume"
TOP = 10
search_xys = []


class WriteLast(Exception):
    pass


def main():
    global search_xys
    search_xys_file = os.path.join(RESUME_DIR, "search_xys.pickle")

    if os.path.exists(search_xys_file):
        search_xys = pickler.load(search_xys_file)
    else:
        raise ValueError()

    user_input = ""

    while not user_input.startswith("quit"):
        user_input = input("enter next search (part,layer(,backtance)|axis:target_value,..): ")

        if not user_input.startswith("quit"):
            query = None

            try:
                part, layer, backtance, query = parse(user_input)
                print("(%s, %s, %s, %s)" % (part, layer, backtance, query))
            except WriteLast as e:
                with open("result-q10.csv", "w") as fh:
                    writer = csv_writer(fh)

                    for r in result[:q10]:
                        writer.writerow(r)

                with open("result-q25.csv", "w") as fh:
                    writer = csv_writer(fh)

                    for r in result[:q25]:
                        writer.writerow(r)

                with open("result-q50.csv", "w") as fh:
                    writer = csv_writer(fh)

                    for r in result[:q50]:
                        writer.writerow(r)
            except Exception as e:
                print(e)
                print("error interpreting: %s" % user_input)

            if query is not None:
                result, q10, q25, q50 = find_closest(part, layer, backtance, query)
                print("found %d: " % len(result))
                print("distance, index, word, expectation, prediction, sentence")

                for r in result[:TOP]:
                    distance, prediction, expectation, backtance, word, index, sequence, point = r
                    print("%s, %s, %s, %s, %s, %s" % (distance, prediction, backtance, word, index, sequence))
        else:
            # Exit path - don't do anything
            pass

    return 0


def parse(user_input):
    if user_input == "write-last":
        raise WriteLast()

    context, targets = user_input.split("|")
    contexts = context.split(",")

    if len(contexts) == 2:
        part, layer = contexts
        backtance = None
    else:
        part, layer, backtance = contexts
        backtance = int(backtance)

    layer = int(layer)
    query = []

    for t in targets.split(","):
        axis, target = t.split(":")
        axis = int(axis)
        target = float(target)
        query += [(axis, target)]

    return part, layer, backtance, query


def find_closest(query_part, query_layer, query_backtance, query):
    assert len(query) > 0, "empty query - would simply return everything!"
    global search_xys
    result = []
    minimum_distance = None
    maximum_distance = None

    for candidate in search_xys:
        xy, prediction, part, layer, index, point = candidate

        if query_part == part and query_layer == layer:
            sub_point = [point[axis] for axis, _ in query]
            target_point = [target for axis, target in query]
            distance = geometry.distance(sub_point, target_point)
            backtance = len(xy.x) - index - 1

            if query_backtance is None or query_backtance == backtance:
                result += [(distance, prediction, xy.y, backtance, xy.x[index], index, xy.x, point)]

                if minimum_distance is None or distance < minimum_distance:
                    minimum_distance = distance

                if maximum_distance is None or distance > maximum_distance:
                    maximum_distance = distance

    if len(result) == 0:
        return result

    q50 = maximum_distance * .5
    q25 = maximum_distance * .25
    q10 = maximum_distance * .1
    print("distance stats: [%.4f, %.4f] q50: %.4f q25: %.4f q10: %.4f" % (minimum_distance, maximum_distance, q50, q25, q10))
    sorted_result = sorted(result)
    cut10 = int(len(result) * 0.1)
    cut25 = int(len(result) * 0.25)
    cut50 = int(len(result) * 0.5)

    histogram_q10 = {"negative": {}, "neutral": {}, "positive": {}}
    for r in sorted_result[:cut10]:
        if r[3] not in histogram_q10[r[1]]:
            histogram_q10[r[1]][r[3]] = 0

        histogram_q10[r[1]][r[3]] += 1
        q10 = r[0]

    histogram_q25 = {"negative": {}, "neutral": {}, "positive": {}}
    for r in sorted_result[:cut25]:
        if r[3] not in histogram_q25[r[1]]:
            histogram_q25[r[1]][r[3]] = 0

        histogram_q25[r[1]][r[3]] += 1
        q25 = r[0]

    histogram_q50 = {"negative": {}, "neutral": {}, "positive": {}}
    for r in sorted_result[:cut50]:
        if r[3] not in histogram_q50[r[1]]:
            histogram_q50[r[1]][r[3]] = 0

        histogram_q50[r[1]][r[3]] += 1
        q50 = r[0]

    print("q50: %.4f q25: %.4f q10: %.4f" % (q50, q25, q10))
    print("histograms\n  q50: %s\n  q25: %s\n  q10: %s" % (adjutant.dict_as_str(rollup(histogram_q50)), adjutant.dict_as_str(rollup(histogram_q25)), adjutant.dict_as_str(rollup(histogram_q10))))
    print("histograms\n  q50: %s\n  q25: %s\n  q10: %s" % (adjutant.dict_as_str(histogram_q50), adjutant.dict_as_str(histogram_q25), adjutant.dict_as_str(histogram_q10)))
    return sorted_result, cut10, cut25, cut50


def rollup(d):
    return {k: sum([item[1] for item in v.items()]) for k, v in d.items()}


if __name__ == "__main__":
    main()

