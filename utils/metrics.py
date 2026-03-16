from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs" / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def _sobel_kernels(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    kernel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]], device=device
    ).unsqueeze(0)
    kernel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]], device=device
    ).unsqueeze(0)
    return kernel_x, kernel_y


def handcrafted_features(images: torch.Tensor) -> torch.Tensor:
    images = images.float()
    if images.dim() == 3:
        images = images.unsqueeze(1)
    pooled = torch.nn.functional.avg_pool2d(images, kernel_size=4).flatten(1)
    kernel_x, kernel_y = _sobel_kernels(images.device)
    edge_x = torch.nn.functional.conv2d(images, kernel_x, padding=1)
    edge_y = torch.nn.functional.conv2d(images, kernel_y, padding=1)
    edge_mag = torch.sqrt(edge_x.pow(2) + edge_y.pow(2) + 1e-8)
    edge_stats = torch.stack(
        [edge_mag.mean(dim=(1, 2, 3)), edge_mag.std(dim=(1, 2, 3)), images.mean(dim=(1, 2, 3)), images.std(dim=(1, 2, 3))],
        dim=1,
    )
    return torch.cat([pooled, edge_stats], dim=1)


def prototype_quality_score(real_images: torch.Tensor, generated_images: torch.Tensor) -> dict[str, float]:
    real = handcrafted_features(real_images)
    fake = handcrafted_features(generated_images)
    mean_distance = torch.norm(real.mean(0) - fake.mean(0), p=2).item()
    variance_distance = torch.norm(real.var(0) - fake.var(0), p=2).item()
    combined = mean_distance + 0.5 * variance_distance
    score = 1.0 / (1.0 + combined)
    return {
        "feature_distance": combined,
        "quality_score": score,
    }


def save_image_grid(
    images: torch.Tensor, path: Union[str, Path], title: Optional[str] = None, nrow: int = 4
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(images.detach().cpu(), nrow=nrow, normalize=True, value_range=(-1, 1))
    fig = plt.figure(figsize=(6, 6))
    plt.axis("off")
    if title:
        plt.title(title)
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def format_quality_note() -> str:
    return "This is a prototype-level quality evaluation based on handcrafted feature distance."
