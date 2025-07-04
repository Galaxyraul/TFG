import torch
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from infovae import InfoVAE, info_vae_loss
import os
import argparse 

parser = argparse.ArgumentParser(description='Model trainer')
parser.add_argument('--dataset','-d',type=str,required=True,help='dataset name')
parser.add_argument('--epochs','-e',type=int,default=20,help='number of epochs')
args = parser.parse_args()
dataset_name = args.dataset

# ==== Config ====
BATCH = 128
Z = 128
LR = 1e-4
EPOCHS = args.epochs
SIZE = 64
LAMBDA = 50.0
ALPHA = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Create Folders ====
os.makedirs(f"infovae/reconstructions/{dataset_name}", exist_ok=True)
os.makedirs(f"infovae/checkpoints/{dataset_name}", exist_ok=True)

# ==== Model and Optimizer ====
model = InfoVAE(img_channels=3, latent_dim=Z).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
fixed_noise = torch.randn(8, model.latent_dim).to(device)

# ==== DataLoader ====
transform = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.CenterCrop(SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
train_loader = DataLoader(
    datasets.ImageFolder(root=f'../../../data/{dataset_name}', transform=transform),
    batch_size=BATCH,
    shuffle=True
)

# ==== Training Loop ====
for epoch in range(EPOCHS):
    model.train()
    total_loss = total_recon_loss = total_kl = total_mmd = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for x, _ in loop:
        x = x.to(device)

        x_recon, mu, logvar, z = model(x)
        loss, recon, kl, mmd = info_vae_loss(x, x_recon, mu, logvar, z, ALPHA, LAMBDA)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_recon_loss += recon.item()
        total_kl += kl.item()
        total_mmd += mmd.item() * LAMBDA

        loop.set_postfix({
            "Loss": f"{loss.item() / len(x):.2f}",
            "Recon": f"{recon.item() / len(x):.2f}",
            "KL": f"{kl.item() / len(x):.2f}",
            "MMD": f"{mmd.item() * LAMBDA / len(x):.4f}"
        })

    # ==== Evaluation ====
    model.eval()
    with torch.no_grad():
        x_sample, _ = next(iter(train_loader))
        x_sample = x_sample.to(device)[:8]
        recon, _, _, _ = model(x_sample)

        # Generate new samples from prior
        generated = model.decoder(fixed_noise)

        comparison = torch.cat([x_sample.cpu(), recon.cpu(), generated.cpu()])
        torchvision.utils.save_image(
            torchvision.utils.make_grid(comparison, nrow=8),
            f"infovae/reconstructions/{dataset_name}/recons_epoch_{epoch+1:03d}.png",
        )

    # ==== Save Checkpoint ====
    torch.save(model.state_dict(),f"infovae/checkpoints/{dataset_name}/infovae_epoch_{epoch+1:03d}.pt")

    # ==== Logging ====
    dataset_size = len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Loss: {total_loss / dataset_size:.4f} | "
          f"Recon: {total_recon_loss / dataset_size:.4f} | "
          f"KL: {total_kl / dataset_size:.4f} | "
          f"MMD: {total_mmd / dataset_size:.4f}")
