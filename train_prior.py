import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
    npimg = img.numpy()
    fig = plt.imshow( np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.savefig('output_gen.png')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)




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

"""Training params"""
learning_rate = 3e-4
batch_size = 256
epochs = 20


optimizer = Adam(prior.parameters(), lr=learning_rate)


training_loader = DataLoader(training_data, 
                             batch_size=batch_size, 
                             shuffle=True,
                             pin_memory=True)

val_dataloader = DataLoader(validation_data,
                             batch_size=batch_size,
                             shuffle=False,
                             pin_memory=True)

# validation
def validate(loader, model, prior, device, num_embeddings, best_loss):
    loss_sum = 0
    prior.eval()
    with torch.no_grad():
        i=0
        for x, label in loader:
            i+=1
            enc = model.encode_int(x.to(device)).squeeze(1)
            logits = prior(enc, label.to(device)).permute(0,2,3,1).contiguous()

            loss_sum += F.cross_entropy(logits.view(-1, num_embeddings), enc.view(-1)).item()
    loss = loss_sum/i

    print(f"Val loss: {loss}")
    if loss < best_loss:
        labels = torch.randint(10, (32,)).int().to(device)
        latents_int = prior.sample(labels, device).long()

        enc = model.vq.to_embedding(latents_int)
        valid_reconstructions = model.reconstruct(enc)
        show(make_grid(torch.clamp(valid_reconstructions.cpu().data+0.5, min=0.0,max=1.0) ))

        return True, loss
    return False, best_loss


best_loss = 10000000000000
model.eval()

"""Training"""
for epoch in range(epochs):
    pbar = tqdm(
                    training_loader,
                    unit="batches",
                    ascii=True,
                    dynamic_ncols=True
                )
    pbar.set_description(f"Running epoch {epoch+1}/{epochs}")
    loss_epoch = 0

    for idx, (x, label) in enumerate(pbar):
        batch_size = x.shape[0]
        # compute actual latents (quantized, with indices - no dense)
        with torch.no_grad():
            x = x.to(device)
            latent = model.encode_int(x).detach().squeeze(1)

        label = label.to(device)
        # forward
        logits = prior(latent, label).permute(0,2,3,1).contiguous()
        
        optimizer.zero_grad()

        # cross entropy between output prior and actual latents
        loss = F.cross_entropy(logits.view(-1, num_embeddings), latent.view(-1))

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        tr = {}
        tr.update({"loss" : loss_epoch/(idx+1)})
        pbar.set_postfix(**tr)
    
    # save best model
    save, best_loss = validate(val_dataloader, model, prior, device, num_embeddings, best_loss)
    if save:
        torch.save(prior.state_dict(), "./vqvae_prior.pt")

    prior.train()