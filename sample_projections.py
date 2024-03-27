import matplotlib.pyplot as plt
import kdtree
import numpy as np

mynames = ['donut', 'onion', 'sheep', 'octopus']
n_to_plot = 1000
# projected_points = np.load("projections.npy")
projected_points = np.load("projections_tsne.npy")
target_x_ratio = 350
target_y_ratio = 280
target_distance_squared = 64

min_x, max_x = np.min(projected_points[:,0]), np.max(projected_points[:,0])
min_y, max_y = np.min(projected_points[:,1]), np.max(projected_points[:,1])

projected_points[:,0] = target_x_ratio * (projected_points[:,0] - min_x) / (max_x - min_x)
projected_points[:,1] = target_y_ratio * (projected_points[:,1] - min_y) / (max_y - min_y)

named_samples = dict()
named_samples_coords = dict()

tree = kdtree.create(dimensions=2)
for i, name in enumerate(mynames):
    samples = []
    data = projected_points[i*n_to_plot:(i+1)*n_to_plot]
    shuffled_indicies = np.random.permutation(n_to_plot)
    j = 0 
    point = data[shuffled_indicies[j]]
    samples.append(shuffled_indicies[j])
    tree.add((point[0], point[1]))
    for j in np.arange(1, n_to_plot):
        point = data[shuffled_indicies[j]]
        _neighbour, distance_2 = tree.search_nn(point)
        if distance_2 > target_distance_squared:
            samples.append(shuffled_indicies[j])
            tree.add((point[0], point[1]))
    named_samples[name] = samples
    named_samples_coords[f"{name}_coords"] = data[[samples],:2][0]

np.savez(
    file="final_projections.npz",
    mynames=np.array(mynames),
    n_to_plot = 1000,
    **named_samples,
    **named_samples_coords,
)

plt.figure()
# plt.scatter(data[:,0], data[:,1], s=0.1)
for i, name in enumerate(mynames):
    data = projected_points[i*n_to_plot:(i+1)*n_to_plot]
    samples = named_samples[name]
    plt.scatter(data[[samples],0], data[[samples],1], label=name)
plt.legend()
plt.savefig('visTSNE_downsampled.png')
plt.close()
        
