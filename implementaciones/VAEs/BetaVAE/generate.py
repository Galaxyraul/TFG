import os
import torch
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm

# === Config ===
model_path = '/raid/rgt00024/tfg/from_scratch/vaes/betavae/checkpoints/drones/beta_vae_epoch100.pt'       # Your pretrained model (.pth)
output_dir = 'generated_drones_seed_33/'     # Where to save the images
num_images = 5000                        # How many images to generate
seed = 33                                # Fixed seed for reproducibility
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# === Load model ===
from betavae import BetaVAE    # Replace with your actual model class
model = BetaVAE().to(device)
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
        img = model.decode(z, class_labels) if class_labels is not None else model.decode(z)

    # Save using save_image with normalization
    save_path = os.path.join(output_dir, f"{i:06d}.png")
    save_image(img, save_path, normalize=True, value_range=(-1, 1))

print("✅ All images generated and saved.")
