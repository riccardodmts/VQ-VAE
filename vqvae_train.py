"""Training for VQ-VAE, CIFAR-10"""

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from models.vqvae import VQVAE
from torch.optim import Adam
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torch
import numpy as np

import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def train(dataloader, vae_model, opt, variance_data, num_iter, device):

    """train fucntion for VQ-VAE"""

    vae_model.train()
    recon_losses = []

    for i in range(num_iter):
        x, _ = next(iter(dataloader))
        x = x.to(device)
        opt.zero_grad()

        # forward + commitment loss
        x_recon, commitment_loss = vae_model(x)

        # reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)/variance_data

        loss = recon_loss + commitment_loss

        loss.backward()
        opt.step()

        recon_losses.append(recon_loss.item())

        if (i+1) % 100 == 0:
            print('%d iterations' % (i+1))
            print('recon_error: %.3f' % np.mean(recon_losses[-100:]))
            print()

training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))

validation_data = datasets.CIFAR10(root="data", train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))

data_variance = np.var(training_data.data / 255.0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""PARAMS"""
batch_size = 256
num_training_updates = 15000

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

learning_rate = 1e-3


training_loader = DataLoader(training_data, 
                             batch_size=batch_size, 
                             shuffle=True,
                             pin_memory=True)


model = VQVAE(num_hiddens, num_residual_hiddens, num_residual_layers, num_embeddings, embedding_dim, commitment_cost, device=device).to(device)

optimizer = Adam(model.parameters(), lr=learning_rate, amsgrad=False)

train(training_loader, model, optimizer, data_variance, num_training_updates, device)

"""SAVE MODEL"""
torch.save(model.state_dict(), "./vqvae_model.pt")

#model.load_state_dict(torch.load("./vqvae_model.pt"))

"""SHOW RESULTS: reconstructions for validations data"""

validation_loader = DataLoader(validation_data,
                               batch_size=32,
                               shuffle=True,
                               pin_memory=True)

model.eval()

(valid_originals, _) = next(iter(validation_loader))
valid_originals = valid_originals.to(device)

# encode samples
enc_output = model.encode(valid_originals)
#enc_output = model.vq.sample((32,8, 8)) # try this to see that the model is not able to generate samples -> suitable prior needed

# reconstruct
valid_reconstructions = model.reconstruct(enc_output)


"""PLOT"""

def show(img):
    npimg = img.numpy()
    fig = plt.imshow( np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.savefig('output.png')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

# clamp needed if output out of range [0,1]
show(make_grid(torch.clamp(valid_reconstructions.cpu().data+0.5, min=0.0,max=1.0) ))