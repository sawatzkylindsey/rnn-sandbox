
from csv import writer as csv_writer
import os
import pdb

from nnwd import geometry
from nnwd import pickler
from pytils import adjutant


RESUME_DIR = ".resume"
TOP = 10
train_xys = []
validation_xys = []
test_xys = []



class WriteLast(Exception):
    pass


def main():
    global train_xys
    global validation_xys
    global test_xys
    train_xys_file = os.path.join(RESUME_DIR, "xys.train.pickle")
    validate_xys_file = os.path.join(RESUME_DIR, "xys.validation.pickle")
    test_xys_file = os.path.join(RESUME_DIR, "xys.test.pickle")

    if os.path.exists(train_xys_file):
        train_xys = pickler.load(train_xys_file)
        validate_xys = pickler.load(validate_xys_file)
        test_xys = pickler.load(test_xys_file)
    else:
        raise ValueError()

    user_input = ""

    while not user_input.startswith("quit"):
        user_input = input("enter next search (dataset|word,..): ")

        if not user_input.startswith("quit"):
            query = None

            try:
                dataset, query = parse(user_input)
                print("(%s, %s, %s)" % (dataset, query))
            except WriteLast as e:
                with open("data.csv", "w") as fh:
                    writer = csv_writer(fh)

                    for r in result:
                        writer.writerow(r)
            except Exception as e:
                print(e)
                print("error interpreting: %s" % user_input)

            if query is not None:
                result = find_closest(dataset, query)
                print("found %d: " % len(result))
                print("distance, sentence, expectation")

                for r in result[:TOP]:
                    # (distance, index, word, xy.x, xy.y, prediction, point)
                    print(r)
        else:
            # Exit path - don't do anything
            pass

    return 0


def parse(user_input):
    if user_input == "write-last":
        raise WriteLast()

    dataset, targets = user_input.split("|")
    query = []

    for t in targets.split(","):
        query += [t.split(" ")]

    return dataset, query


def find_closest(dataset, query):
    assert len(query) > 0, "empty query - would simply return everything!"
    global train_xys
    global validation_xys
    global test_xys
    xys = train_xys if dataset == "train" else (validation_xys if dataset == "validation" else test_xys)
    result = []
    histogram = {"negative": 0, "neutral": 0, "positive": 0}

    for candidate in xys:
        sequence = candidate[0]
        matches = 0

        for target in query:
            i = 0
            j = 0
            maximum_i = len(sequence) - len(target) + 1
            resume = None
            found = False

            while not found and i < maximum_i:
                if target[j] == sequence[i]:
                    if resume is None:
                        resume = i + 1

                    if j == len(target) - 1:
                        found = True
                    else:
                        j += 1
                        i += 1
                else:
                    j = 0

                    if resume is not None:
                        i = resume
                        resume = None
                    else:
                        i += 1

            if found:
                matches += 1

        distance = len(query) - matches

        if distance == 0:
            histogram[candidate[1]] += 1

        if matches > 0:
            result += [(distance, candidate[1], candidate[0])]

    sorted_result = sorted(result)
    print("histogram @0: %s" % (adjutant.dict_as_str(histogram)))
    return sorted_result


if __name__ == "__main__":
    main()

