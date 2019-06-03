import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as coll


step = 5
count = int(255 / 5)


for r in range(count):
    figure = plt.figure()
    ax = plt.subplot(111, aspect='equal')
    ax.axis([0, count, 0, count])
    for g in range(count):
        for b in range(count):
            color = "#%02x%02x%02x" % ((r * step) + 2, (g * step) + 2, (b * step) + 2)
            print(color)
            sq = patches.Rectangle((g, b), step, step, color=color)
            ax.add_patch(sq)
    #plt.draw()
    #plt.pause(120)
    figure.savefig("grid-%d.png" % r)



