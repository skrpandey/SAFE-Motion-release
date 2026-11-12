"""TinyTCN: no-graph temporal convolution baseline (pilot protocol v1, batch 2).

Joints are flattened into channels, so the model has no skeleton topology. Input (N, C, T, V).
"""

import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, c, dilation, p_drop=0.1):
        super().__init__()
        self.conv = nn.Conv1d(c, c, 5, padding=2 * dilation, dilation=dilation)
        self.bn = nn.BatchNorm1d(c)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        return x + self.drop(self.act(self.bn(self.conv(x))))


class TinyTCN(nn.Module):
    def __init__(self, c_in=5, num_joints=17, width=96, dilations=(1, 2, 4, 8), num_classes=2):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(c_in * num_joints)
        self.stem = nn.Sequential(nn.Conv1d(c_in * num_joints, width, 1),
                                  nn.BatchNorm1d(width), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[ResBlock(width, d) for d in dilations])
        self.fc = nn.Linear(width, num_classes)

    def forward_features(self, x):
        n, c, t, v = x.shape
        x = self.input_bn(x.permute(0, 1, 3, 2).reshape(n, c * v, t))
        return self.blocks(self.stem(x)).mean(dim=2)

    def forward(self, x):
        return self.fc(self.forward_features(x))
