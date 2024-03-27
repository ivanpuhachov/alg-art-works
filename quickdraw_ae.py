
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pathlib import Path
from tqdm import tqdm


class QDdataset(Dataset):
    def __init__(
            self,
            folder="quickdraw_data/",
            names=['donut'],
            n_per_name=1000,
            transform=None,
            return_label=False,
            ) -> None:
        super().__init__()
        numpy_list = []
        for name in names:
            npdata = np.load(Path(folder) / f"{name}.npy")[:n_per_name] / 255
            numpy_list.append(npdata.reshape(n_per_name, 28, 28))
        self._items = np.concatenate(numpy_list, axis=0)
        mean = np.mean(self._items)
        std = np.std(self._items)
        # self._items -= mean
        # self._items /= std
        self.transform = transform
        self.n_per_name = n_per_name
        self.return_label = return_label
    
    def __len__(self):
        return self._items.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # if self.transform:
        #     sample = self.transform(sample)
        if self.return_label:
            return torch.from_numpy(self._items[idx]).float().unsqueeze(0), idx // self.n_per_name
        else:
            return torch.from_numpy(self._items[idx]).float().unsqueeze(0)


class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1,16,kernel_size=5, padding=2),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 8, kernel_size=2, stride=2),
            torch.nn.Conv2d(8, 1, kernel_size=3, padding=1),
            # torch.nn.Dropout(p=0.2),
            # torch.nn.ReLU(True),
            # torch.nn.Conv2d(16, 8, kernel_size=2, stride=2),
            # torch.nn.Conv2d(8, 8, kernel_size=3, padding=1),
            # torch.nn.Dropout(p=0.2),
            # torch.nn.ReLU(True),
            # torch.nn.Conv2d(8, 8, kernel_size=2, stride=1),
            # torch.nn.Conv2d(8, 4, kernel_size=1),
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.Linear(14*14,49),
            # torch.nn.Conv2d(1, 6, kernel_size=5),
            # torch.nn.ReLU(True),
            # torch.nn.Dropout(p=0.2),
            # torch.nn.Conv2d(6,16,kernel_size=5),
            # torch.nn.ReLU(True),
            # torch.nn.Conv2d(16, 16, kernel_size=2, stride=2),
            # torch.nn.ReLU(True),
            # torch.nn.Conv2d(16, 16, kernel_size=3, stride=1),
            # torch.nn.Conv2d(16, 4, kernel_size=1),
            # torch.nn.ReLU(True),
            # torch.nn.Flatten(start_dim=1, end_dim=-1),
            # torch.nn.Linear(64, 16),
            )

        self.decoder = torch.nn.Sequential(  
            torch.nn.Linear(49,14*14),
            torch.nn.ReLU(),
            torch.nn.Linear(14*14,14*14),
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.2),
            # torch.nn.ReLU(True),
            # torch.nn.Linear(64,256),
            # torch.nn.ReLU(True),
            torch.nn.Dropout(p=0.2),
            torch.nn.Unflatten(1, (4,7,7)),
            torch.nn.Conv2d(4, 32, kernel_size=3, padding=1),
            torch.nn.PixelShuffle(upscale_factor=2),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(8, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, kernel_size=3, padding=1),
            # torch.nn.ReLU(True),
            # torch.nn.PixelShuffle(upscale_factor=2),
            # torch.nn.Conv2d(4, 4, kernel_size=3, padding=1),
            # torch.nn.ReLU(True),
            # torch.nn.Dropout(p=0.2),
            # torch.nn.Conv2d(4, 4, kernel_size=3),
            # torch.nn.ReLU(True),
            # torch.nn.Conv2d(4, 1, kernel_size=3, padding=1),
            # torch.nn.ReLU(True),
            # torch.nn.Sigmoid(),
            )
    
    def latent(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    qd = QDdataset(
        names=['baseball', 'donut', 'lollipop', 'moon'],
        n_per_name=400,
        )
    dataloader = DataLoader(qd, batch_size=100,
                        shuffle=True, num_workers=0,)
    model = Autoencoder().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-2, weight_decay=1e-5)

    model.eval()
    with torch.no_grad():
        dd = next(iter(dataloader)).cuda()
        print(dd.shape)
        ee = model.latent(dd)
        print(ee.shape)
        oo = model(dd)
        print(oo.shape)
        # raise NotImplementedError
    plt.figure()
    plt.subplot(121)
    plt.imshow(dd[0,0].detach().cpu().numpy())
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(oo[0,0].detach().cpu().numpy())
    plt.colorbar()
    plt.savefig("init.png")
    plt.close()
    # plt.show()

    best_loss = 1000
    for epoch in tqdm(range(1000)):
        model.train()
        total_loss = 0
        for it, data in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data.cuda())
            loss = torch.nn.MSELoss()(output, data.cuda())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # pbar.set_description(f"{total_loss:.5f}")
        if total_loss < 0.9 * best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), "best.ckpt")
        # print(f"epoch {epoch} : {total_loss:.5f}")
    
    model.load_state_dict(torch.load("best.ckpt"))
    model.eval()
    dd = next(iter(dataloader)).cuda()
    print(dd.shape)
    oo = model(dd)
    print(oo.shape)
    plt.figure()
    plt.subplot(121)
    plt.imshow(dd[0,0].detach().cpu().numpy())
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(oo[0,0].detach().cpu().numpy())
    plt.colorbar()
    plt.savefig("fin.png")
    plt.close()
    # plt.show()
        
        