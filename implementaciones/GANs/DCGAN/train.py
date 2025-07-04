import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets,transforms
from torchvision.utils import save_image,make_grid
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
from dcgan import Discriminator,Generator,initialize_weights
from tqdm import tqdm
import os
import argparse 

parser = argparse.ArgumentParser(description='Model trainer')
parser.add_argument('--dataset','-d',type=str,required=True,help='dataset name')
parser.add_argument('--epochs','-e',type=int,default=20,help='number of epochs')
args = parser.parse_args()
dataset_name = args.dataset

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    exit()

os.makedirs(f'dcgan/samples/{dataset_name}',exist_ok=True)

LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS = 3
Z_DIM = 128
NUM_EPOCHS = args.epochs
FEATURES_DISC = 64
FEATURES_GEN = 64

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    #transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(CHANNELS)],[0.5 for _ in range(CHANNELS)]
    )
])

"""dataset = datasets.MNIST(
    root='dataset/',
    train=True,
    transform=transforms,
    download=True
)"""

dataset = datasets.ImageFolder(root=f'../../data/{dataset_name}',
                               transform=transform)

dataloader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)
gen = Generator(Z_DIM,CHANNELS,FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS,FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(),lr=LEARNING_RATE,betas=(0.5,0.999))
opt_disc = optim.Adam(disc.parameters(),lr=LEARNING_RATE,betas=(0.5,0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32,Z_DIM,1,1).to(device)
#writer_real = SummaryWriter(f"logs/real")
#writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()

for epoch in tqdm(range(NUM_EPOCHS),total=NUM_EPOCHS,desc='Epoch:'):
    for batch_idx,(real,_) in tqdm(enumerate(dataloader),total=len(dataloader),desc='Batch'):
        real = real.to(device)
        noise = torch.randn((BATCH_SIZE,Z_DIM,1,1)).to(device)
        fake = gen(noise)
        
        ### Train Disc
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real,torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)
        loss_disc_fake = criterion(disc_fake,torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_fake + loss_disc_real)/2
        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()
        
        ### Train GEN
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output,torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()        
    
    with torch.no_grad():
        fake = gen(fixed_noise)
        comparison = torch.cat([real[:8].cpu(), fake[:8].cpu()])
        save_image(make_grid(comparison, nrow=8 ),
        f"dcgan/samples/{dataset_name}/epoch_{epoch+1:03d}.png")
    os.makedirs(f'dcgan/checkpoints/{dataset_name}/{epoch+1}',exist_ok=True)
    torch.save(gen.state_dict(), f"dcgan/checkpoints/{dataset_name}/{epoch+1}/gen.pt")
    torch.save(disc.state_dict(), f"dcgan/checkpoints/{dataset_name}/{epoch+1}/disc.pt")

