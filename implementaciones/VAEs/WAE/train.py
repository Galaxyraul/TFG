import torch
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from wae import WAE,mmd_loss
import os
import argparse 

parser = argparse.ArgumentParser(description='Model trainer')
parser.add_argument('--dataset','-d',type=str,required=True,help='dataset name')
parser.add_argument('--epochs','-e',type=int,default=20,help='number of epochs')
args = parser.parse_args()
dataset_name = args.dataset

os.makedirs(f"reconstructions/{dataset_name}", exist_ok=True)
os.makedirs(f"checkpoints/{dataset_name}", exist_ok=True)
# ---- Config ----
BATCH = 16
Z = 128
LR = 1e-4
EPOCHS = args.epochs
SIZE = 64
LAM = 150.0
if torch.cuda.is_available():
    device = 'cuda'
else:
    exit()

# Initialize model, critic, and optimizer
model = WAE(img_channels=3, latent_dim=Z).to(device)
optimizer_model = optim.Adam(model.parameters(), lr=LR)
fixed_noise = torch.randn(8, model.latent_dim).to(device)

transform = transforms.Compose([
    transforms.Resize((SIZE,SIZE)),
    transforms.CenterCrop(SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
# DataLoader
train_loader = DataLoader(datasets.ImageFolder(root=f'../../../data/{dataset_name}', transform=transform), batch_size=64, shuffle=True)

# Training Loop
for epoch in range(EPOCHS):
    model.train()

    total_loss = 0
    total_recon_loss = 0
    total_mmd_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for x, _ in loop:
        x = x.to(device)

        x_recon, z = model(x)
        z_prior = torch.randn_like(z)

        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
        mmd = mmd_loss(z, z_prior)
        total_loss_batch = recon_loss + LAM * mmd

        optimizer_model.zero_grad()
        total_loss_batch.backward()
        optimizer_model.step()

        total_loss += total_loss_batch.item()
        total_recon_loss += recon_loss.item()
        total_mmd_loss += mmd.item()

        loop.set_postfix({
            "Loss": f"{total_loss_batch.item():.2f}",
            "Recon": f"{recon_loss.item():.2f}",
            "MMD": f"{mmd.item():.2f}"
        })

    # === Evaluation and Saving ===
    model.eval()
    with torch.no_grad():
        x_sample, _ = next(iter(train_loader))
        x_sample = x_sample.to(device)[:8]
        recon, _ = model(x_sample)

        z_random = torch.randn(8, model.latent_dim).to(device)
        generated = model.decode(fixed_noise)

        comparison = torch.cat([x_sample.cpu(), recon.cpu(), generated.cpu()])
        torchvision.utils.save_image(
            torchvision.utils.make_grid(comparison, nrow=8),
            f"reconstructions/{dataset_name}/recons_epoch_{epoch+1:03d}.png"
        )

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_model_state_dict': optimizer_model.state_dict(),
    }, f"checkpoints/{dataset_name}/wae_epoch_{epoch+1:03d}.pt")


    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader.dataset):.2f}, Recon Loss: {total_recon_loss/len(train_loader.dataset):.2f}")
