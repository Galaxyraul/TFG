import torch
from torch import nn
import torch.nn.functional as F

class InfoVAE(nn.Module):
    def __init__(self, img_channels=3,latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1),  # 64 → 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),           # 32 → 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),          # 16 → 8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),         # 8 → 4
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4 → 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 8 → 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 16 → 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),  # 32 → 64
            nn.Tanh() 
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z

def compute_mmd(x, y):
    xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2 * xx
    dyy = ry.t() + ry - 2 * yy
    dxy = rx.t() + ry - 2 * xy

    # Use multiple bandwidths
    scales = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    mmd = 0
    for scale in scales:
        K = torch.exp(-dxx / (2 * scale))
        L = torch.exp(-dyy / (2 * scale))
        P = torch.exp(-dxy / (2 * scale))
        mmd += K.mean() + L.mean() - 2 * P.mean()

    return mmd / len(scales)

def info_vae_loss(x, x_recon, mu, logvar, z, alpha=0.5, lambda_mmd=100.0):
    recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)

    # KL divergence
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    # Sample from prior
    z_prior = torch.randn_like(z)
    mmd = compute_mmd(z, z_prior)
    
    # Correct loss per InfoVAE formulation
    total_loss = recon_loss + (1 - alpha) * kl + (lambda_mmd + alpha - 1) * mmd
    return total_loss, recon_loss, kl, mmd

if __name__=='__main__':
    z = torch.randn(64, 128)
    z_prior = z + 3.0
    print("MMD:", compute_mmd(z, z_prior).item())  # Should be ~0.5 or higher
