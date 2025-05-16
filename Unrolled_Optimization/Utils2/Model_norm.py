import torch
import torch.nn as nn

# ==============================================================
# ConvBlock
# ==============================================================
class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels    : int,
                 out_channels   : int,
                 kernel_size    : int = 3,
                 stride         : int = 2,
                 padding        : int = 2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

# ==============================================================
# DeconvBlock
# ==============================================================
class DeconvBlock(nn.Module):
    def __init__(self,
                 in_channels    : int,
                 out_channels   : int,
                 kernel_size    : int = 2,
                 stride         : int = 2,
                 padding        : int = 0,
                 output_padding : int = 0):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               output_padding=output_padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.deconv(x)

class TinyUNET(nn.Module):
    def __init__(self):
        super().__init__()
        # Input normalization
        self.norm = nn.InstanceNorm2d(2)

        # Encoder
        self.e1 = ConvBlock(2, 64, padding=1)
        self.e2 = ConvBlock(64, 128, padding=1)
        self.e3 = ConvBlock(128, 256, padding=1)
        self.e4 = ConvBlock(256, 512, padding=1)

        # Decoder
        self.d1 = DeconvBlock(512, 256)
        self.d2 = DeconvBlock(256, 128)
        self.d3 = DeconvBlock(128, 64)
        self.d4 = DeconvBlock(64, 2)  # Output 2 channels

        # Skip connections
        self.c1 = ConvBlock(512, 256, stride=1, padding=1)
        self.c2 = ConvBlock(256, 128, stride=1, padding=1)
        self.c3 = ConvBlock(128, 64, stride=1, padding=1)
        self.c4 = nn.Conv2d(4, 2, kernel_size=3, padding=1)  # Final layer

    def forward(self, x_complex):
        # Handle complex input [B,1,H,W]
        x = torch.cat([x_complex.real, x_complex.imag], dim=1)  # [B,2,H,W]
        x = self.norm(x)

        # Encoder
        e1 = self.e1(x)  # [B,64,H/2,W/2]
        e2 = self.e2(e1)  # [B,128,H/4,W/4]
        e3 = self.e3(e2)  # [B,256,H/8,W/8]
        e4 = self.e4(e3)  # [B,512,H/16,W/16]

        # Decoder
        d1 = self.d1(e4)  # [B,256,H/8,W/8]
        c1 = self.c1(torch.cat([d1, e3], dim=1))

        d2 = self.d2(c1)  # [B,128,H/4,W/4]
        c2 = self.c2(torch.cat([d2, e2], dim=1))

        d3 = self.d3(c2)  # [B,64,H/2,W/2]
        c3 = self.c3(torch.cat([d3, e1], dim=1))

        d4 = self.d4(c3)  # [B,2,H,W]
        output = self.c4(torch.cat([d4, x], dim=1))  # [B,2,H,W]

        # Recombine to complex
        return torch.complex(output[:, 0:1], output[:, 1:2])  # [B,1,H,W]