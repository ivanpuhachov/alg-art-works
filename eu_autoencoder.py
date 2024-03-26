import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.manifold import TSNE

import seaborn as sns
import pandas as pd


from quickdraw_ae import QDdataset


# mnist_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
mnist_data = QDdataset(names=['donut', 'onion', 'sheep', 'octopus'])
mnist_data = list(mnist_data)[:4096]

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encoder_forward(self, x):
        return self.encoder(x)

    # decode reduced dimensionality data
    def decoder_forward(self, x):
        return self.decoder(x)

def train(model, num_epochs=50, batch_size=64, learning_rate=1e-3):
    model = model.cuda()
    torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5) # <--
    train_loader = torch.utils.data.DataLoader(mnist_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    outputs = []
    for epoch in range(num_epochs):
        for data in train_loader:
            img, _ = data
            img = img.cuda()
            recon = model(img.cuda())
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
        outputs.append((epoch, img.detach().cpu(), recon.detach().cpu()),)
    return outputs


model = Autoencoder()
max_epochs = 50
outputs = train(model, num_epochs=max_epochs)

for k in range(0, max_epochs, 5):
    plt.figure(figsize=(9, 2))
    imgs = outputs[k][1].detach().numpy()
    recon = outputs[k][2].detach().numpy()
    for i, item in enumerate(imgs):
        if i >= 9: break
        plt.subplot(2, 9, i + 1)
        plt.imshow(item[0], cmap="coolwarm")

    for i, item in enumerate(recon):
        if i >= 9: break
        plt.subplot(2, 9, 9 + i + 1)
        plt.imshow(item[0])
    plt.show()

train_loader = torch.utils.data.DataLoader(mnist_data,
                                           batch_size=10,
                                           shuffle=False)
output_matrix = []
keys_t = []
model = model.cpu()
with torch.no_grad():
    output_matrix = []
    for data in train_loader:
        x, keys = data
        x_encoded = model.encoder_forward(x)
        output_matrix.append(x_encoded)
        keys_t.append(keys)
    output_matrix = torch.cat(output_matrix, dim=0).squeeze(2).squeeze(2)
    keys_t = torch.cat(keys_t, dim=0)
# output_matrix = torch.cat(output_matrix)
print(output_matrix.shape)

plt.figure()
plt.title("ORIGINAL")
plt.scatter(output_matrix[:, 0], output_matrix[:, 1])
plt.show()

X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(output_matrix)

np.save("projections_tsne.npy", arr=X_embedded)

data = pd.DataFrame(X_embedded, columns=['tsne_0','tsne_1'])
data['key'] = keys_t
print(data.shape)
# shapesns.regplot(x=X_embedded[:,0], y=X_embedded[:,1],label = keys_t)
# plt.show()
sns.scatterplot(data=data, x="tsne_0", y="tsne_1", hue="key", alpha = 0.4)
plt.show()
