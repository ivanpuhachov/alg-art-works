import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import umap

reducer = umap.UMAP()

data = np.load("latents_baseball.npy")
data2 = np.load("latents_donut.npy")
data3 = np.load("latents_moon.npy")
print(data.shape)

mynames = ['octopus', 'bird', 'donut', 'lollipop', 'moon', 'rabbit', 'sheep', 'onion', 'owl']

n_to_plot = 1000
alldata = np.concatenate(
    [
        np.load(f"latents_{name}.npy")[:n_to_plot]
        for name in mynames
    ], 
    axis=0)
print("Computing tSNE")
tsne = TSNE(n_components=2)
projections = tsne.fit_transform(alldata)

np.save("projections_tsne.npy", arr=projections)

plt.figure()
for i in range(len(mynames)):
    plt.scatter(projections[i*n_to_plot:(i+1)*n_to_plot,0], projections[i*n_to_plot:(i+1)*n_to_plot,1], alpha=0.6, label=mynames[i])
plt.legend()
plt.savefig("visTSNE.png")
plt.close()

print("Computing UMAP")
embedding = reducer.fit_transform(alldata)
print(embedding.shape)

plt.figure()
for i in range(len(mynames)):
    plt.scatter(embedding[i*n_to_plot:(i+1)*n_to_plot,0], embedding[i*n_to_plot:(i+1)*n_to_plot,1], alpha=0.6, label=mynames[i])
plt.legend()
plt.savefig("visUMAP.png")
plt.close()

np.save("umaps.npy", arr=embedding)

# reduced_data = projections[:n_to_plot]
# n_clust = 20
# kmeans = KMeans(init="k-means++", n_clusters=n_clust, n_init=4, random_state=0)
# kmeans.fit(reduced_data)
# plt.figure()
# for i in range(n_clust):
#     plt.scatter(reduced_data[kmeans.labels_==i][:,0], reduced_data[kmeans.labels_==i][:,1], alpha=0.6)
# # plt.legend()
# plt.savefig("visTSNE_clust.png")
# plt.close()