
import random
import sys


def randbool():
    return random.choice([True, False])


for i in range(int(sys.argv[1])):
    level = 0
    terminal = True
    sentence = []

    while True:
        if len(sentence) > 0 and terminal and randbool():
            print(" ".join(sentence))
            break
        else:
            if randbool():
                sentence += ["%d" % level]

            if level < 4 and randbool():
                sentence += ["("]
                level += 1

            if level > 0 and randbool():
                sentence += [")"]
                level -= 1

            terminal = level == 0

