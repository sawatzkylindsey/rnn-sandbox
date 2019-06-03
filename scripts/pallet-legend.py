import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as coll


x = y = np.linspace(-1, 1, 21)
z = np.array([i*i+j*j for j in y for i in x])
Z = z.reshape(21, 21)
Z = np.array([
    [[252,141,89]],
    [[255,255,191]],
    [[145,207,96]],
])
print(Z)
plt.imshow(Z, interpolation='bilinear')
plt.show()

#   negative: fc8d59 / 252,141,89
#    neutral: ffffbf / 255,255,191
#   positive: 91cf60 / 145,207,96

#negative: fc8d59 / 252,141,89
#neutral: ffffbf / 255,255,191
#positive: 91cf60 / 145,207,96


#figure = plt.figure()
#axis = figure.add_subplot(111, projection="2d")
#axis.set_xlabel("x")
#axis.set_ylabel("y")

#to_hex = lambda c: "#" + "".join([format(val, "02x") for val in c])
#plot_point = lambda p: list(p) + ([] if dimensions == 3 else [0])
#colouring = lambda x: to_hex([0, min((x * 5) + 100, 255), 0])

#plt.scatter([[0, 1]], [[0, 1]], c=[to_hex([0, 0, 255]), to_hex([255, 0, 0])])

#plt.draw()
#plt.pause(120)
#figure = plt.figure()
#ax = plt.subplot(111, aspect='equal')
#ax.axis([0, count, 0, count])
#    for g in range(count):
#        for b in range(count):
#            color = "#%02x%02x%02x" % ((r * step) + 2, (g * step) + 2, (b * step) + 2)
#            print(color)
#            sq = patches.Rectangle((g, b), step, step, color=color)
#            ax.add_patch(sq)
#    #plt.draw()
#    #plt.pause(120)
#    figure.savefig("legend-%d.png" % r)
#


