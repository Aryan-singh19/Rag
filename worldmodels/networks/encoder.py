import torch
import torch.nn as nn
import einops as op


class ConvEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, x):
        '''
        Extract 64*64*3 image into R^1024

        (B, T, C, H, W) -> (B, T, 1024)    
        '''
        B = x.shape[0]
        x = op.rearrange(x, "B T C H W -> (B T) C H W")
        x = self.net(x)
        x = op.rearrange(x, "(B T) D -> B T D", B=B)
        return x


