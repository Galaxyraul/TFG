import torch
import torch.nn as nn
import torch.nn.functional as F

class WAE(nn.Module):
    def __init__(self, img_channels=3,latent_dim=64):
        super(WAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: Conv -> Latent
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1),  # 64 → 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),           # 32 → 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),          # 16 → 8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),         # 8 → 4
            nn.ReLU(),
        )

        self.fc_z = nn.Linear(256*4*4, latent_dim)

        self.decoder_input = nn.Linear(latent_dim,256*4*4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4 → 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 8 → 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 16 → 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),  # 32 → 64
            nn.Tanh() 
        )

    def encode(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        z = self.fc_z(result)
        return z

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 256, 4, 4)
        result = self.decoder(result)
        return result

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

def compute_kernel(x, y):
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]

    tiled_x = x.view(x_size,1,dim).repeat(1, y_size,1)
    tiled_y = y.view(1,y_size,dim).repeat(x_size, 1,1)

    return torch.exp(-torch.mean((tiled_x - tiled_y)**2,dim=2)/dim*1.0)


def mmd_loss(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2*torch.mean(xy_kernel)