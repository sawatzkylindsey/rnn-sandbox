
import tensorflow as tf
import numpy as np


i1 = tf.placeholder(dtype=tf.int32, shape=[2, 3, 4])
i2 = tf.placeholder(dtype=tf.int32, shape=[3, 2])
#index = tf.Variable(-1)
#def fnn(x):
#    tf.assign_add(index, 1)
#    return [x, index]
#m = tf.map_fn(fnn, i2)
g = tf.gather_nd(i1, i2)

session = tf.Session()

data = [
    [
        [1,2,3,4],
        [2,3,4,5],
        [3,4,5,6]
    ],
    [
        [7,8,9,10],
        [8,9,10,11],
        [9,10,11,12]
    ]
]
print("input:\n%s" % np.array(data))
l = [1, 1, 0]
expected = []

for i, position in enumerate(l):
    expected.append(data[position][i])

print("expected:\n%s" % np.array(expected))

l = [[1, 0], [1, 1], [0, 2]]
print("output:\n%s" % session.run(g, feed_dict={i1: np.array(data), i2: np.array(l)}))

