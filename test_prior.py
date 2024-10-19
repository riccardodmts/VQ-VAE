import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from models.vqvae import VQVAE
from models.pixelcnn import PixelCNN2d
from torch.optim import Adam
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torch
import numpy as np
from tqdm import tqdm


import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def show(img):
    """Save image"""
    npimg = img.numpy()
    fig = plt.imshow( np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.savefig('output_gen.png')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)


validation_data = datasets.CIFAR10(root="data", train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))

val_dataloader = DataLoader(validation_data,
                             batch_size=32,
                             shuffle=False,
                             pin_memory=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""VQ-VAE params"""

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2
embedding_dim = 64
num_embeddings = 512
commitment_cost = 0.25

"""Models"""
model = VQVAE(num_hiddens, num_residual_hiddens, num_residual_layers, num_embeddings, embedding_dim, commitment_cost, device=device).to(device)
model.load_state_dict(torch.load("./vqvae_model.pt"))

prior = PixelCNN2d(input_dim=num_embeddings, dim=64, num_blocks=12).to(device)
prior.load_state_dict(torch.load("./vqvae_prior.pt"))

model.eval()
prior.eval()

#labels = torch.randint(10, (32,)).int().to(device)
labels = torch.ones(32).long().to(device) * 8
print(labels)

latents_int = prior.sample(labels, device).long()
enc = model.vq.to_embedding(latents_int)

valid_reconstructions = model.reconstruct(enc)
show(make_grid(torch.clamp(valid_reconstructions.cpu().data+0.5, min=0.0,max=1.0) ))