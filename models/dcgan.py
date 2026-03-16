from __future__ import annotations

import math
from typing import Union

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 64, base_channels: int = 32) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels

        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels * 4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            nn.Conv2d(base_channels, 1, 3, 1, 1),
            nn.Tanh(),
        )
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(module: nn.Module) -> None:
        classname = module.__class__.__name__
        if "Conv" in classname:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias.data, 0.0)
        elif "BatchNorm" in classname:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0.0)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.net(noise)

    def sample(self, batch_size: int, device: Union[torch.device, str]) -> torch.Tensor:
        noise = torch.randn(batch_size, self.latent_dim, 1, 1, device=device)
        return self.forward(noise)


class Discriminator(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, 2, 0, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4 * 3 * 3, 1),
            nn.Sigmoid(),
        )
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(module: nn.Module) -> None:
        classname = module.__class__.__name__
        if "Conv" in classname or "Linear" in classname:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias.data, 0.0)
        elif "BatchNorm" in classname:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0.0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.features(images)
        return self.classifier(feats)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.features(images)
        return feats.view(feats.size(0), -1)


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def estimate_dense_size_mb(module: nn.Module) -> float:
    total_bytes = 0
    for parameter in module.state_dict().values():
        total_bytes += parameter.numel() * parameter.element_size()
    return total_bytes / math.pow(1024, 2)
