import torch
from torch import nn

class ResidualBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, input_dim, kernel_size=1,
                      stride=1, bias=False),
        )

    def forward(self, x):
        return x + self.net(x)
    
class Encoder(nn.Module):

    def __init__(self, hidden_dim, residual_dim, num_res_blocks):
        super().__init__()
        self.num_res_blocks = num_res_blocks

        self.conv = nn.Sequential(
            nn.Conv2d(3, hidden_dim//2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim//2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        )
        self.residuals = nn.ModuleList(ResidualBlock(hidden_dim, residual_dim) for _ in range(num_res_blocks))

    def forward(self, x):
        x = self.conv(x)

        for i in range(self.num_res_blocks):
            x = self.residuals[i](x)

        return x
    
class Decoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, residual_dim, num_res_blocks):
        super().__init__()

        self.num_res_blocks = num_res_blocks

        self.conv = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.residuals = nn.ModuleList(ResidualBlock(hidden_dim, residual_dim) for _ in range(num_res_blocks))

        self.tr_conv = nn.Sequential(nn.ConvTranspose2d(hidden_dim, hidden_dim//2, kernel_size=4, stride=2, padding=1),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(hidden_dim//2, 3, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):

        x = self.conv(x)
        for i in range(self.num_res_blocks):
            x = self.residuals[i](x)

        return self.tr_conv(x)
                                     

    


        