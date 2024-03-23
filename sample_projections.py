import matplotlib.pyplot as plt
import kdtree
import numpy as np

mynames = ['octopus', 'bird', 'donut', 'lollipop', 'moon', 'rabbit', 'sheep', 'onion', 'owl']
n_to_plot = 1000
# projected_points = np.load("projections.npy")
projected_points = np.load("umaps.npy")


named_samples = dict()

for i, name in enumerate(mynames):
    samples = []
    data = projected_points[i*n_to_plot:(i+1)*n_to_plot]
    shuffled_indicies = np.random.permutation(n_to_plot)
    tree = kdtree.create(dimensions=2)
    j = 0 
    point = data[shuffled_indicies[j]]
    samples.append(shuffled_indicies[j])
    tree.add((point[0], point[1]))
    for j in np.arange(1, n_to_plot):
        point = data[shuffled_indicies[j]]
        _neighbour, distance_2 = tree.search_nn(point)
        if distance_2 > 1:
            samples.append(shuffled_indicies[j])
            tree.add((point[0], point[1]))
    named_samples[name] = samples

plt.figure()
# plt.scatter(data[:,0], data[:,1], s=0.1)
for i, name in enumerate(mynames):
    data = projected_points[i*n_to_plot:(i+1)*n_to_plot]
    samples = named_samples[name]
    plt.scatter(data[[samples],0], data[[samples],1], label=name)
plt.legend()
plt.savefig('visUMAP_downsampled.png')
plt.close()
        
