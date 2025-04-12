import torch
import torch.nn as nn
import torch.nn.functional as F

class EmberDualCNN(nn.Module):
    def __init__(self):
        super(EmberDualCNN, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),  # Batch normalization
                nn.LeakyReLU(),
                nn.MaxPool2d(2),  # Halve the spatial dimensions
                nn.Dropout2d(0.1)  # Dropout to prevent overfitting
            )

        self.byteentropy_branch = nn.Sequential(
            conv_block(1, 32),  # 16x16 → 8x8
            conv_block(32, 64),  # 8x8 → 4x4
            conv_block(64, 128)  # 4x4 → 2x2
        )
        self.histogram_branch = nn.Sequential(
            conv_block(1, 32),  # 16x16 → 8x8
            conv_block(32, 64),  # 8x8 → 4x4
            conv_block(64, 128)  # 4x4 → 2x2
        )

        self.flatten = nn.Flatten()
        self.feature_dim = 128 * 2 * 2 * 2  # 1024 (two branches concatenated)

    def forward(self, byteentropy, histogram):
        x1 = self.byteentropy_branch(byteentropy)
        x2 = self.histogram_branch(histogram)

        x1 = self.flatten(x1)
        x2 = self.flatten(x2)

        return torch.cat((x1, x2), dim=1)  # (B, 1024)
