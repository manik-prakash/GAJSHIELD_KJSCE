import torch
import torch.nn as nn
import torch.nn.functional as F

class EmberDualCNN(nn.Module):
    def __init__(self):
        super(EmberDualCNN, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2),  # Halve spatial dimensions
                nn.Dropout2d(0.2)
            )

        # Byteentropy branch
        self.byteentropy_branch = nn.Sequential(
            conv_block(1, 32),  # 16x16 → 8x8
            conv_block(32, 64),  # 8x8 → 4x4
            conv_block(64, 128)  # 4x4 → 2x2
        )

        # Histogram branch
        self.histogram_branch = nn.Sequential(
            conv_block(1, 32),  # 16x16 → 8x8
            conv_block(32, 64),  # 8x8 → 4x4
            conv_block(64, 128)  # 4x4 → 2x2
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 2 * 2 * 2, 1)  # Binary classification output

    def forward(self, byteentropy, histogram):
        x1 = self.byteentropy_branch(byteentropy)
        x2 = self.histogram_branch(histogram)

        x1 = self.flatten(x1)
        x2 = self.flatten(x2)

        combined_features = torch.cat((x1, x2), dim=1)  # (B, 1024)
        output = self.fc(combined_features)  # (B, 1)
        return output