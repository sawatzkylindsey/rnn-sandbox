
import sys

from nnwd import pickler


with open(sys.argv[1], "r") as fh:
    words = set()
    sequences = []

    for line in fh.readlines():
        if line.strip() != "":
            sequence = []

            for word in line.strip().split(" "):
                sequence += [(word, None)]

                if word not in words:
                    words.add(word)

            sequences += [sequence]

    pickler.dump(sequences, sys.argv[2])
    pickler.dump([word for word in words], "words")

