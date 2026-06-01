import torch
import torch.nn as nn


class SharpenAddNet_deeper_cGAN_v1(nn.Module):
    def __init__(self, N=256):
        super().__init__()
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(3, N, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(N, N, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(N, N, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.encoder_4 = nn.Sequential(
            nn.Conv2d(N, N, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.encoder_5 = nn.Conv2d(N, N, kernel_size=3, stride=2, padding=1)

        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(N, N, kernel_size=3, stride=2, output_padding=1, padding=1),
            nn.GELU(),
        )
        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(N, N, kernel_size=3, stride=2, output_padding=1, padding=1),
            nn.GELU(),
        )
        self.decoder_3 = nn.Sequential(
            nn.ConvTranspose2d(N, N, kernel_size=3, stride=2, output_padding=1, padding=1),
            nn.GELU(),
        )
        self.decoder_4 = nn.Sequential(
            nn.ConvTranspose2d(N, N, kernel_size=3, stride=2, output_padding=1, padding=1),
            nn.GELU(),
        )
        self.decoder_5 = nn.ConvTranspose2d(
            N, 3, kernel_size=3, stride=2, output_padding=1, padding=1
        )

    def forward(self, x):
        x_1 = self.encoder_1(x)
        x_2 = self.encoder_2(x_1)
        x_3 = self.encoder_3(x_2)
        x_4 = self.encoder_4(x_3)
        x_5 = self.encoder_5(x_4)

        x_6 = self.decoder_1(x_5)
        x_7 = self.decoder_2(x_6 + x_4)
        x_8 = self.decoder_3(x_7 + x_3)
        x_9 = self.decoder_4(x_8 + x_2)
        x_10 = self.decoder_5(x_9 + x_1)
        return x_5, x_10
