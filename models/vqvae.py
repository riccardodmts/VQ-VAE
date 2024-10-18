from typing import Mapping
import torch
from torch import nn
import torch.nn.functional as F
from .vqema import VQEMA
from .nets import Encoder, Decoder


class VQVAE(nn.Module):


    def __init__(self, hidden_dim, res_dim, num_residuals, num_embeddings, dim_embeddings, beta=0.25, gamma=0.99, device=None):
        """
        Args:
            hidden_dim: number of channels hidden layers.
            res_dim: number of channels residual blocks.
            num_residuals: number of residual blocks.
            num_embeddings: number of codewords in the codebook e.g. K.
            dim_embeddings: length of any codeword. 
            beta: gain for commitment loss.
            gamma: decaying factor for EMA.
            device:
        """
        super().__init__()

        self.encoder = Encoder(hidden_dim, res_dim, num_residuals)
        self.conv_to_vq = nn.Conv2d(hidden_dim, dim_embeddings, stride=1, kernel_size=1)

        self.vq = VQEMA(num_embeddings, dim_embeddings, beta, gamma, device=device)

        self.decoder = Decoder(dim_embeddings, hidden_dim, res_dim, num_residuals)

    def forward(self, x):
        "Used for training"
        enc = self.encoder(x)
        enc = self.conv_to_vq(enc)
        enc_quantized, vq_loss, _ = self.vq(enc)
        x_recon = self.decoder(enc_quantized)

        return x_recon, vq_loss
    
    """EVAL methods"""

    def reconstruct(self, quantized):
        return self.decoder(quantized)
    
    def encode(self, x):
        """Encode image. NOTE: the latent is a dense vector"""
        enc = self.conv_to_vq(self.encoder(x))
        enc_quantized, _, _ = self.vq(enc)

        return enc_quantized
    
    def encode_int(self, x):
        """Encode image. NOTE: the latent is not dense -> it reutns the indeces of the codewords"""
        enc = self.conv_to_vq(self.encoder(x))
        _, _, enc_idxs = self.vq(enc)

        return enc_idxs
    

    

