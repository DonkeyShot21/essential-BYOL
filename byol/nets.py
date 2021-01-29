import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.l1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.l2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, arch, hidden_dim, proj_dim, low_res):
        super().__init__()

        # backbone
        self.encoder = models.__dict__[arch]()
        self.feat_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Identity()

        # modify the encoder for lower resolution
        if low_res:
            self.encoder.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.encoder.maxpool = nn.Identity()
            self._reinit_all_layers()

        # build heads
        self.projection = MLP(self.feat_dim, hidden_dim, proj_dim)

    @torch.no_grad()
    def _reinit_all_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feats = self.encoder(x)
        z = self.projection(feats)
        return z, feats