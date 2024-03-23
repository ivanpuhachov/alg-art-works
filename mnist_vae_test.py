"""
https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb
"""

# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from quickdraw_ae import QDdataset
from torch.utils.data import Dataset, DataLoader
import numpy as np

bs = 1000
my_z_dim = 16
# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

# print(next(iter(test_loader))[0][0])
# raise NotImplementedError

mynames = ['octopus', 'bird', 'donut', 'lollipop', 'moon', 'rabbit', 'sheep', 'onion', 'owl']
figs_per_name = 8000

qd = QDdataset(
        names=mynames,
        n_per_name=figs_per_name,
        )

qdataloader = DataLoader(qd, batch_size=bs,
                        shuffle=True, num_workers=0,)


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        
        # encoder part
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=2)
        # self.conv2 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1)
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        # self.fc2_ = nn.Linear(h_dim2, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        # self.fc4_ = nn.Linear(h_dim2, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        # self.conv_1 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, padding=1)
        
    def encoder(self, x):
        # hh = torch.unflatten(x, dim=1, sizes=(1,28,28))
        # hh = F.relu(self.conv1(hh))
        # hh = F.relu(self.conv2(hh))
        # # hh = F.relu(self.conv3(hh))
        # h = torch.flatten(hh, start_dim=1)
        # h = F.relu(self.fc1(h))
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        # h = F.relu(self.fc2_(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        # h = F.relu(self.fc4_(h))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h)) 
        # h = F.sigmoid(self.fc6(h)) 
        # hh = torch.unflatten(input=h, dim=1, sizes=(4,14,14))
        # hh = F.upsample(input=hh, scale_factor=2, mode='bilinear')
        # hh = F.relu(self.conv_1(hh))
        # return F.relu(torch.flatten(input=hh, start_dim=1))
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

# build model
vae = VAE(x_dim=28*28, h_dim1= 1024, h_dim2=512, z_dim=my_z_dim)
if torch.cuda.is_available():
    vae.cuda()

optimizer = optim.Adam(vae.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50,)
# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # L2 = F.l1_loss(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, data in enumerate(qdataloader):
        data = data.reshape(-1, 28*28).cuda()
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = vae(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        scheduler.step()
        
        # if batch_idx % 100 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(qdataloader.dataset),
        #         100. * batch_idx / len(qdataloader), loss.item() / len(data)))
    print('->Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(qdataloader.dataset)))

def test():
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for data in qdataloader:
            data = data.reshape(-1,28*28).cuda()
            recon, mu, log_var = vae(data)
            
            # sum up batch loss
            test_loss += loss_function(recon, data, mu, log_var).item()
        
    test_loss /= len(qdataloader.dataset)
    print('-> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

for epoch in range(1, 1001):
    train(epoch)
    if epoch %10 == 9:
        test()

torch.save(vae.state_dict(), "vae.ckpt")

# vae.load_state_dict(torch.load("vae.ckpt"))

with torch.no_grad():
    z = torch.randn(64, my_z_dim).cuda()
    sample = vae.decoder(z).cuda()
    
    save_image(sample.view(64, 1, 28, 28), 'sample_' + '.png')

with torch.no_grad():
    for nn in mynames:
        qdsmall = QDdataset(names=[nn], n_per_name=figs_per_name)
        dl = DataLoader(qdsmall, batch_size=bs,
                        shuffle=False, num_workers=0,)
        latents = []
        for dd in dl:
            data = dd.reshape(-1,28*28).cuda()
            recon, mu, log_var = vae(data)
            latents.append(
                mu.detach().cpu().numpy()
            )
        print(latents[0].shape)
        print(latents[-1].shape)
        print(len(latents))
        print("----")
        latents = np.concatenate(latents, axis=0)
        np.save(file=f'latents_{nn}.npy', arr=latents)