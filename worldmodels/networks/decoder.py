import torch
import torch.nn as nn
import einops as op

class ConvDecoder(nn.Module):
    def __init__(self, feat_dim) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.proj = nn.Linear(in_features=feat_dim, out_features=1024)

        self.net = nn.Sequential(
                nn.ConvTranspose2d(in_channels=1024, out_channels=128, 
                                   kernel_size=5, stride=2, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=128, out_channels=64, 
                                   kernel_size=5, stride=2, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=64, out_channels=32, 
                                   kernel_size=6, stride=2, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=3, 
                                   kernel_size=6, stride=2, padding=0),
            )
        
    def forward(self, x:torch.Tensor):
        '''
        Decode concatenated (s_t, h_t) features into 64x64 RGB image means.

        (B, T, 230) -> (B, T, C, H, W)    
        '''
        B = x.shape[0]
        x = op.rearrange(x, "B T D -> (B T) D")
        x = self.proj(x)
        B_and_T = x.shape[0]
        C = x.shape[1]
        x = x.reshape([B_and_T, C, 1 , 1])
        x = self.net(x)
        x = op.rearrange(x, "(B T) C H W -> B T C H W", B=B)
        return x
        
