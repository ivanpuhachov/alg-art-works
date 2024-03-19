import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

data = np.load("latents_baseball.npy")
data2 = np.load("latents_donut.npy")
data3 = np.load("latents_moon.npy")
print(data.shape)

n_to_plot = 500
alldata = np.concatenate([data[:n_to_plot], data2[:n_to_plot], data3[:n_to_plot]], axis=0)
tsne = TSNE(n_components=2, random_state=0)
projections = tsne.fit_transform(alldata)

plt.figure()
plt.scatter(projections[:n_to_plot,0], projections[:n_to_plot,1], alpha=0.6)
plt.scatter(projections[n_to_plot:2*n_to_plot,0], projections[n_to_plot:2*n_to_plot,1], alpha=0.6)
plt.scatter(projections[2*n_to_plot:,0], projections[2*n_to_plot:,1], alpha=0.6)
plt.savefig("visTSNE.png")
plt.close()

# ix, iy = 4, 2
# plt.figure()
# plt.scatter(data[:,ix], data[:,iy], alpha=0.6)
# plt.scatter(data2[:,ix], data2[:,iy], alpha=0.6)
# plt.scatter(data3[:,ix], data3[:,iy], alpha=0.6)
# plt.savefig("vis.png")
# plt.close()
# plt.show()