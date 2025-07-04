import torch
from torch import nn
import torch.nn.functional as F

class BetaVAE(nn.Module):
    def __init__(self,img_channels = 3,latent_dim=128,beta = 4.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels,32,4,2,1),
            nn.ReLU(),
            nn.Conv2d(32,64,4,2,1),
            nn.ReLU(),
            nn.Conv2d(64,128,4,2,1),
            nn.ReLU(),
            nn.Conv2d(128,256,4,2,1),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        self.fc_z = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),  # 32x32 -> 64x64
            nn.Tanh(),  # Output range [0, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self,x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self,z):
        h = self.fc_z(z)
        h = h.view(h.size(0), 256, 4, 4)
        return self.decoder(h)

    def forward(self,x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


    def loss_function(self, recon, x, mu, logvar):
        # Reconstruction loss (pixel-wise BCE)
        recon_loss = F.mse_loss(recon, x, reduction='sum')

        # KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        return recon_loss + self.beta * kl_div, recon_loss, kl_div


if __name__ == '__main__':
    # Instantiate the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BetaVAE(img_channels=3, latent_dim=128, beta=4.0).to(device)

    # Generate a batch of random input (e.g., batch_size=8, 3x64x64 image)
    batch_size = 64
    random_input = torch.randn(batch_size, 3, 64, 64).to(device) 

    # Forward pass
    recon, mu, logvar = model(random_input)

    # Print shapes to confirm
    print(f"Input shape:        {random_input.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Latent mean shape:    {mu.shape}")
    print(f"Latent logvar shape:  {logvar.shape}")

    # Compute loss
    loss, recon_loss, kl_div = model.loss_function(recon, random_input, mu, logvar)
    print(f"Total loss: {loss.item():.2f}, Recon Loss: {recon_loss.item():.2f}, KL Div: {kl_div.item():.2f}")

