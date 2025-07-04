import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from betavae import BetaVAE  # assuming your model is in vae.py
import os
import argparse 

parser = argparse.ArgumentParser(description='Model trainer')
parser.add_argument('--dataset','-d',type=str,required=True,help='dataset name')
parser.add_argument('--epochs','-e',type=int,default=20,help='number of epochs')
args = parser.parse_args()
dataset_name = args.dataset

# ---- Config ----
BATCH = 128
Z = 128
BETA = 4.0
LR = 1e-4
EPOCHS = args.epochs 
SIZE = 64
if torch.cuda.is_available():
    device = 'cuda'
else:
    exit()

os.makedirs(f"betavae/reconstructions/{dataset_name}", exist_ok=True)
os.makedirs(f"betavae/checkpoints/{dataset_name}", exist_ok=True)

# ---- DataLoader (Use CelebA or dummy data for now) ----
transform = transforms.Compose([
    transforms.Resize((SIZE,SIZE)),
    transforms.CenterCrop(SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Replace with actual CelebA when available
train_dataset = datasets.ImageFolder(
    root=f'../../data/{dataset_name}',
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)

# ---- Model ----
model = BetaVAE(img_channels=3, latent_dim=Z, beta=BETA).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
fixed_noise = torch.randn(8, model.latent_dim).to(device)

# ---- Training Loop ----
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for x, _ in loop:
        x = x.to(device)

        # Forward
        recon, mu, logvar = model(x)

        # Compute loss (MSE-based)
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + BETA * kl_div

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_div.item()
        loop.set_postfix({
            "Loss": f"{loss.item() / len(x):.2f}",
            "Recon": f"{recon_loss.item() / len(x):.2f}",
            "KL": f"{kl_div.item() / len(x):.2f}"
        })


    model.eval()
    with torch.no_grad():
        x_sample, _ = next(iter(train_loader))
        x_sample = x_sample.to(device)[:8]
        recon, _, _ = model(x_sample)
        samples = model.decode(fixed_noise)

    comparison = torch.cat([x_sample.cpu(), recon.cpu(),samples.cpu()])  # Originals on top, recon on bottom

    save_image(make_grid(comparison, nrow=8 ),
            f"betavae/reconstructions/{dataset_name}/epoch_{epoch+1:03d}.png")
    print(f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Total Loss: {total_loss/len(train_loader.dataset):.2f} | "
        f"Recon: {total_recon/len(train_loader.dataset):.2f} | "
        f"KL: {total_kl/len(train_loader.dataset):.2f}")

    # Optional: Save model
    torch.save(model.state_dict(), f"betavae/checkpoints/{dataset_name}/beta_vae_epoch{epoch+1}.pt")
