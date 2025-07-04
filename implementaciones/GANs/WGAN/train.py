import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets,transforms
from torchvision.utils import save_image,make_grid
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
from wgan import Critic,Generator,initialize_weights
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

os.makedirs(f'wgan/samples/{dataset_name}',exist_ok=True)

LEARNING_RATE = 2e-5
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS = 3
Z_DIM = 128
NUM_EPOCHS = args.epochs
FEATURES_DISC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01

transforms = transforms.Compose([
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

dataset = datasets.ImageFolder(root=f'../../data/{dataset_name}',transform=transforms)

dataloader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)
gen = Generator(Z_DIM,CHANNELS,FEATURES_GEN).to(device)
critic = Critic(CHANNELS,FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.RMSprop(gen.parameters(),lr=LEARNING_RATE)
opt_critic = optim.RMSprop(critic.parameters(),lr=LEARNING_RATE)


fixed_noise = torch.randn(32,Z_DIM,1,1).to(device)
#writer_real = SummaryWriter(f"logs/real")
#writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
critic.train()

for epoch in tqdm(range(NUM_EPOCHS),total=NUM_EPOCHS,desc='Epoch:'):
    for batch_idx,(real,_) in tqdm(enumerate(dataloader),total=len(dataloader),desc='Batch'):
        real = real.to(device)
        
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn((BATCH_SIZE,Z_DIM,1,1)).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            loss_critic = torch.mean(critic_fake) - torch.mean(critic_real)
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            for p in critic.parameters():
                p.data.clamp_(-WEIGHT_CLIP,WEIGHT_CLIP)

        ### Train generator
        output = critic(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        
    with torch.no_grad():
        fake = gen(fixed_noise)
        comparison = torch.cat([real[:8].cpu(), fake[:8].cpu()])
        save_image(make_grid(comparison, nrow=8 ),
        f"wgan/samples/{dataset_name}/epoch_{epoch+1:03d}.png")
    os.makedirs(f'wgan/checkpoints/{dataset_name}/{epoch+1}',exist_ok=True)
    torch.save(gen.state_dict(), f"wgan/checkpoints/{dataset_name}/{epoch+1}/gen.pt")
    torch.save(critic.state_dict(), f"wgan/checkpoints/{dataset_name}/{epoch+1}/critic.pt")
    

