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

# ==============================================================
# Modified TinyUNET
# ==============================================================

class TinyUNET(nn.Module):
    def __init__(self):
        super(TinyUNET, self).__init__()

        # ========================
        # input is now 2 channels (real + imag)
        # ========================
        self.e1 = ConvBlock(2, 64, padding=1)
        self.e2 = ConvBlock(64, 128, padding=1)
        self.e3 = ConvBlock(128, 256, padding=1)
        self.e4 = ConvBlock(256, 512, padding=1)

        self.d1 = DeconvBlock(512, 256)
        self.d2 = DeconvBlock(256, 128)
        self.d3 = DeconvBlock(128, 64)

        # ========================
        # Output is now 2 channels (real + imag)
        # ========================
        self.d4 = DeconvBlock(64, 2, padding=0)

        # Skip Connections
        self.c1 = ConvBlock(512, 256, stride=1, padding=1)
        self.c2 = ConvBlock(256, 128, stride=1, padding=1)
        self.c3 = ConvBlock(128, 64, stride=1, padding=1)

        # ========================
        # Final output should also be 2 channels
        # ========================
        self.c4 = ConvBlock(4, 2, stride=1, padding=1)

    def forward(self, x_complex):
        # ========================
        # Split real & imag and concatenate
        # ========================
        #print('x_complex',x_complex.shape)
        x = torch.cat([x_complex.real, x_complex.imag], dim=1)  # (B, 2, H, W)
        #print('x',x.shape)
        # Encoder
        e1 = self.e1(x)
        #print('e1',e1.shape)
        e2 = self.e2(e1)
        #print('e2',e2.shape)
        e3 = self.e3(e2)
        #print('e3',e3.shape)
        e4 = self.e4(e3)
        #print('e4',e4.shape)

        # Decoder with skip connections
        d1 = self.d1(e4)
        #print('d1',d1.shape)
        #print('e3',e3.shape)
        c1d = torch.cat([d1, e3], dim=1)
        #print('concat1',c1d.shape)
        c1 = self.c1(c1d)
        #print('c1',c1.shape)

        d2 = self.d2(c1)
        #print('d2',d2.shape)
        #print('e2',e2.shape)
        c2d = torch.cat([d2, e2], dim=1)
        #print('concat2',c2d.shape )
        c2 = self.c2(c2d)
        #print('c2',c2.shape)

        d3 = self.d3(c2)
        #print('d3',d3.shape)
        #print('e1',e1.shape)
        c3d = torch.cat([d3, e1], dim=1)
        #print('concat3',c3d.shape)
        c3 = self.c3(c3d)
        #print('c3',c3.shape)

        d4 = self.d4(c3)
        #print('d4',d4.shape)
        #print('x',x.shape)
        c4d = torch.cat([d4, x], dim=1)  # Input x has 2 channels (real + imag)
        #print('concat4',c4d.shape)
        c4 = self.c4(c4d)
        #print('c4',c4.shape)

        # ========================
        # Reconstruct complex output
        # ========================
        real = c4[:, 0:1]
        #print('real',real.shape)
        imag = c4[:, 1:2]
        #print('imag',imag.shape)
        return torch.complex(real, imag)

# ==============================================================
# Model instance
# ==============================================================
model = TinyUNET()