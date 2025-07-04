import os
import torch
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm

# === Config ===
model_path = '/raid/rgt00024/tfg/from_scratch/vaes/infovae/checkpoints/celeba/infovae_epoch_020.pt'      # Your pretrained model (.pth)
output_dir = 'generated_celeba_seed_33/'     # Where to save the images
num_images = 5000                        # How many images to generate
seed = 33                                # Fixed seed for reproducibility
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# === Load model ===
from infovae import InfoVAE    # Replace with your actual model class
model = InfoVAE(img_channels=3,latent_dim=128).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Set seed
torch.manual_seed(seed)
np.random.seed(seed)

# === Create output folder
os.makedirs(output_dir, exist_ok=True)

# === Generate images
for i in tqdm(range(num_images), desc="Generating"):
    # Sample random latent vector
    z = torch.randn(1,model.latent_dim, device=device)

    # Optional: class labels if needed
    class_labels = None
    if hasattr(model, 'label_dim') and model.label_dim > 0:
        class_labels = torch.zeros([1, model.label_dim], device=device)

    # Forward pass
    with torch.no_grad():
        img = model.decoder(z, class_labels) if class_labels is not None else model.decoder(z)

    # Save using save_image with normalization
    save_path = os.path.join(output_dir, f"{i:06d}.png")
    save_image(img, save_path, normalize=True, value_range=(-1, 1))

print("✅ All images generated and saved.")
