"""
https://stackoverflow.com/questions/17817889/is-there-any-way-to-add-points-to-kd-tree-implementation-in-scipy
"""
import random
import matplotlib.pyplot as plt
import kdtree

SIZE = 100
N = 10_000
MIN_DISTANCE = 5
MIN_DISTANCE_2 = MIN_DISTANCE * MIN_DISTANCE

x, y = random.uniform(0, SIZE), random.uniform(0, SIZE)

tree = kdtree.create(dimensions=2)
tree.add((x, y))

xs, ys = [], []
not_xs, not_ys = [], []

for _ in range(N):
    x, y = random.uniform(0, SIZE), random.uniform(0, SIZE)
    _neighbour, distance_2 = tree.search_nn((x, y))
    if distance_2 > MIN_DISTANCE_2:
        xs.append(x)
        ys.append(y)
        tree.add((x, y))
    else:
        not_xs.append(x)
        not_ys.append(y)

plt.axes().set_aspect(1)
plt.scatter(not_xs, not_ys, color='red', s=0.1)
plt.scatter(xs, ys)
plt.savefig("kdtree.png")
# plt.show()
plt.close()