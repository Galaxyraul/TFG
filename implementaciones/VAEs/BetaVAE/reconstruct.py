import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image

# === Config ===
input_dir = '/raid/rgt00024/tfg/data/drones/drones'              # Folder with original images
output_dir = 'recon_images_drones'             # Output folder
model_path = '/raid/rgt00024/tfg/from_scratch/vaes/betavae/checkpoints/drones/beta_vae_epoch100.pt'        # Your model checkpoint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Load model ===
from betavae import BetaVAE   # Replace with actual model
model = BetaVAE().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Transform: normalize to [-1, 1]
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === Create output dir
os.makedirs(output_dir, exist_ok=True)

# === Process images
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for fname in tqdm(image_files, desc="Reconstructing"):
    img_path = os.path.join(input_dir, fname)
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        x_recon,_,_ = model(x)

    # Save reconstructed image (denormalized automatically by save_image)
    save_image(x_recon, os.path.join(output_dir, fname), normalize=True, value_range=(-1, 1))

print("✅ All reconstructions saved.")
