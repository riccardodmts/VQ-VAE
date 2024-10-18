from typing import Mapping
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class VQEMA(nn.Module):
    """Implementation for EMA update for codewords in the codebook"""

    def __init__(self, num_embeddings, dim_embeddings, commit_gain, gamma, device, epsilon=1e-5):
        """
        Args:
            num_embeddings: number of codewords in the codebook e.g. K.
            dim_embeddings: length of any codeword 
            commit_gain: gain for commitment loss.
            gamma: decaying factor for EMA.
            device:
            epsilon: used for Laplace smoothing.
        """
        super().__init__()

        self.K = num_embeddings # codebook dimension
        self.D = dim_embeddings # embedding dimension
        self.beta = commit_gain # gain for commitment loss e.g. beta * ||sg[e]-z_e(x)||_2
        self.gamma = gamma # deacy factor for EMA
        self.epsilon = epsilon  # used for Laplace smoothing
        self.device = device

        self.register_buffer("ema_m", torch.zeros((num_embeddings, dim_embeddings)).float())  # m vectors keeping exponential sum e.g. m_i^t
        self.ema_m.normal_().to(device)  # initialize them
        self.register_buffer("ema_M", torch.zeros((num_embeddings, 1)).float()) # M values e.g M_i^t
        self.register_buffer("embeddings", torch.zeros_like(self.ema_m))
        
        self.embeddings = self.ema_m.to(device)  # initialize the embeddings (we don't divide by M)
        self.ema_M.to(device)

    def forward(self, enc_pred):

        """
        Args:
            enc_pred (torch.Tensor): encoder output, [B, D, H, W]
        Returns:
            (torch.Tensor): quantized output, [B, D, H, W].
        """
        
        enc_pred = enc_pred.permute(0,2,3,1).contiguous()  # from [B,C,H,W] to [B,H,W,C], channel dimension = embedding dimension (C=D)
        shape = enc_pred.shape

        flat_pred = enc_pred.view(-1, self.D)  # create a view with all prediction of dimension [B * H * W, D]

        """Compute closest embedding per encoder prediction"""

        # clever way to compute distance: as for scalar case (x-y)^2=x^2+y^2-2xy
        distances = (torch.sum(flat_pred**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings**2, dim=1)
                    - 2 * torch.matmul(flat_pred, self.embeddings.t()))

        min_dist_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # for encoder output get the index of closes embedding, [*, 1]
        min_dist_indices_return = min_dist_indices.view(shape[0], shape[1], shape[2], 1).permute(0,3,1,2)


        # trick to get the corresponding embedding for each prediction
        mask_encodings = torch.zeros(min_dist_indices.shape[0], self.K).to(self.device)  # [*, K]
        mask_encodings.scatter_(1, min_dist_indices, 1)  # mask: for each encoder output, a one-hot for the correspoding embedding, [*, K]

        quantized = torch.matmul(mask_encodings, self.embeddings)  # quatized vector for each prediction
        quantized = quantized.view(shape) # back to [B, H, W, C]

        """Update embedding with EMA"""
        with torch.no_grad():
            if self.training:
                # update m
                dm = torch.matmul(mask_encodings.t(), flat_pred)    # this is a smart way to compute the sum -> [K, D]                                                                  
                self.ema_m = self.gamma * self.ema_m + (1-self.gamma) * dm

                # update M
                self.ema_M = self.gamma * self.ema_M + (1-self.gamma) * torch.sum(mask_encodings, dim=0).reshape(-1,1)

                # to avoid division by zero at the next step, Laplace smoothing

                N = torch.sum(self.ema_M)  # total number of trials (number of predictions, but with EMA is not the actual sum of all samples seen)
                self.ema_M = (self.ema_M + self.epsilon) / (N + self.epsilon * self.K) * N # we mulitply by M since this is a sum and not an average

                #update embeddings
                self.embeddings = self.ema_m / self.ema_M         

        """ Compute Loss for commitment """
        loss = F.mse_loss(quantized.detach(), enc_pred)

        """ Straight Through Estimator """
        # clever way to compute STE: the gradient flow using the input for the decoder, namely quantized. To make the gradient flow, we just
        # compute "quantized" as the encoder output minus the difference betweeen "quantized" - the encoder output using detach
        # by doing so we don't have to copy the gradient, but pytorch will do it for us
        # namely, quantization appears as the Jacobian between the decoder input and the encoder input is the identity matrix
        quantized = enc_pred + (quantized - enc_pred).detach()

        """Return"""
        quantized = quantized.permute(0,3,1,2) # back to [B, C, H, W]

        return quantized, self.beta * loss, min_dist_indices_return


    def sample(self, size):
        """not used - use it to sample from uniform"""
        indices = torch.from_numpy(np.random.randint(self.K, size=size, dtype=np.int64)).view(-1, 1)  # [B, H, W] -> [B * H * W, 1]
        mask_encodings = torch.zeros(indices.shape[0], self.K)  # [*, K]
        mask_encodings.scatter_(1, indices, 1)

        quantized = torch.matmul(mask_encodings.to(self.device), self.embeddings)  # [*, D]

        return quantized.view((*size, self.D)).permute(0,3,1,2)
    
    def to_embedding(self, indices):

        """
        Given the latent with integer indices, this function returns the corresponding dense embedding.
        Args:
            indices (torch.Tensor): latent, with indices not the dense embedding. [B, H, W]
        Returns:
            (torch.Tensor): latent with dense vectors
        """
        size = indices.shape
        idxs = indices.view(-1).unsqueeze(-1)
        mask_encodings = torch.zeros(idxs.shape[0], self.K).to(self.device)  # [*, K]
        mask_encodings.scatter_(1, idxs, 1)

        quantized = torch.matmul(mask_encodings.to(self.device), self.embeddings)  # [*, D]

        return quantized.view((*size, self.D)).permute(0,3,1,2)





