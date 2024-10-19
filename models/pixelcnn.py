import torch
from torch import nn
import torch.nn.functional as F

class GatedActivation(nn.Module):

    def __init__(self, dim=1):

        super().__init__()
        self.dim = dim  # dim to create 2 chunks

    def forward(self, x):

        x1, x2 = torch.chunk(x, 2, self.dim)
        return F.tanh(x1) * F.sigmoid(x2)


def freeze_vmask(grad):
    grad[:,:,-1,:] = 0.0

def freeze_hmask(grad):
    grad[:,:,:,-1] = 0.0

class GatedMaskConv2d2(nn.Module):
    """Implementation of Gated Masked Convolution assuming original input has just one channel (no RGB images).
    Masking is just used for spatial dimensions."""

    def __init__(self, num_channels, kernel_size, num_classes, residual=True, mask_type="A"):
        """
        Args:
            num_channels: number of hidden units (channel wise).
            kernel_size: 
            num_classes: number of classes.
            residual: if True, skip connection is used for horizontal stack.
            mask_type: if "A", the filter is causal.
        """
        super().__init__()

        """VSTACK
        To implement masking, instead of using convolutional filters with zeros we use the pad&shift trick
        see: https://sergeiturukin.com/2017/02/24/gated-pixelcnn.html (not completely correct: if you use a kernel 3x3 by doing so,
        we get masked 5x5 filter since we have 2 )
        """
        self.type = mask_type
        self.kernel_size = kernel_size
        self.residual = residual
        
        # we assume kernel size 3x3, 5x5 or 7x7 (in general odd number)
        kernel_size_vstack = (kernel_size//2+1, kernel_size)  # kernel size (n//2, n)
        padding_vstack = (kernel_size//2, kernel_size//2)  # padding

        self.vstack = nn.Conv2d(num_channels, num_channels * 2, kernel_size_vstack, 1, padding_vstack)
        #self.vstack.weight.data[:, :, :, :]= 1.0  #test
        # In my opinion this should be always full of zeros (both A and B)
        self.vstack.weight.data[:, :, -1, :]= 0.0
        self.vstack.weight.register_hook(freeze_vmask)

        self.v_to_h = nn.Conv2d(2*num_channels, 2*num_channels, 1)

        """HSTACK"""
        kernel_size_hstack = (1, kernel_size//2+1)
        padding_hstack = (0, kernel_size//2)

        self.hstack = nn.Conv2d(num_channels, 2*num_channels, kernel_size_hstack, 1, padding_hstack)
        #self.hstack.weight.data[:, :, :, :]= 1.0  #test

        if self.type == "A":
            # if type A, mask current pixel -> mask last column conv filters
            self.hstack.weight.data[:, :, :, -1]= 0.0
            self.hstack.weight.register_hook(freeze_hmask)

        self.h_to_res = nn.Conv2d(num_channels, num_channels, 1)

        # an embedding for each class (instead of one hot encodings)
        self.class_condition = nn.Embedding(num_classes, num_channels * 2)

        # gates
        self.vgate = GatedActivation(dim=1)
        self.hgate = GatedActivation(dim=1)



    def forward(self, x_v, x_h, h):

        h_v = self.vstack(x_v)[:, :, :x_v.shape[-2]]

        h = self.class_condition(h).unsqueeze(-1).unsqueeze(-1)  # from [B, C] to [B, C, 1, 1]
        out_v = self.vgate(h_v + h)

        v_to_h = self.v_to_h(h_v)

        h_h = self.hstack(x_h)[..., :x_v.shape[-1]]
        out_h = self.hgate(h_h + v_to_h + h)

        out_h = self.h_to_res(out_h)

        if self.residual:
            out_h += x_h

        return out_v, out_h
    

class PixelCNN2d(nn.Module):

    """PixelCNN implementation. NOTE: we assume the input has dimension [B, H, W], namely this implementation
    works for 'gray scale' inputs. -> no masking for channel dimension. """


    def __init__(self, dim=64, num_blocks=15, num_classes=10, input_dim=256):
        """
        Args:
            dim: number of hidden units (channel wise).
            num_blocks: number of Gated blocks.
            num_classes: number of classes.
            input_dim: number of codewords in the codebook e.g. K
        """

        super().__init__()

        # map each pixel value from integer [0,input_dim-1] to dense 1d tensor
        self.map_input = nn.Embedding(input_dim, dim)

        self.gated_blocks = nn.ModuleList()

        for i in range(num_blocks):

            kernel_size, mask_type, residual = (7, "A", False) if i==0 else (3, "B", True)

            self.gated_blocks.append(GatedMaskConv2d2(dim, kernel_size, num_classes,
                                                      residual=residual, mask_type=mask_type))
            
        # Output block (just 1x1 conv)

        self.ouput = nn.Sequential(
            nn.Conv2d(dim, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, input_dim, 1)
        )

    def forward(self, x, label):
        """
        Args:
            x (torch.Tensor): target sample e.g. actual quantized latent. [B, H, W]
            label (torch.Tensor, dtype=int): label for the target sample.

        Returns:
            (torch.Tensor): logits for each sample. [B, K, H, W,] K=number of codewords in the codebook.
        """

        B, H, W = x.shape
        x = self.map_input(x.view(-1))  # [B * H * W, D]

        x = x.view(B, H, W, -1)  # [B, H, W, D]
        x = x.permute(0, 3, 1, 2)  # [B, D, H, W]

        x_v, x_h = (x, x)

        for i in range(len(self.gated_blocks)):
            x_v, x_h = self.gated_blocks[i](x_v, x_h, label)

        return self.ouput(x_h)
    

    def sample(self, label, device, shape=(8,8)):

        """
        Args:
            label (torch.Tensor): class to condition p(z). [B]
            device:
            shape (tuple): specify spatial dimension for latent (8x8 for CIFAR-10)

        Returns:
            (torch.Tensor): sample from conditional prior p(z|class). [B, *shape] e.g. [B, 8, 8]
        """

        batch_size = label.shape[0]

        x = torch.zeros((batch_size, *shape)).int().to(device)

        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = self.forward(x, label)
                probs = F.softmax(logits[:,:,i, j], dim=-1)
                samples = torch.multinomial(probs, 1).squeeze().float()
                x[:,i, j] = samples

        return x



