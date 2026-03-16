from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.dcgan import Discriminator, Generator
from utils.io import CHECKPOINTS_DIR, SAMPLES_DIR, ensure_output_dirs, save_checkpoint
from utils.metrics import prototype_quality_score, save_image_grid
from utils.seed import set_seed


def build_dataloader(batch_size: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.FashionMNIST(root=PROJECT_ROOT / "data", train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)


def evaluate_generator(generator: Generator, latent_dim: int, device: torch.device, batch_size: int) -> dict[str, float]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.FashionMNIST(root=PROJECT_ROOT / "data", train=False, download=True, transform=transform)
    real_batch = torch.stack([dataset[index][0] for index in range(batch_size)]).to(device)
    generator.eval()
    with torch.no_grad():
        fake_batch = generator.sample(batch_size, device)
    generator.train()
    return prototype_quality_score(real_batch, fake_batch)


def train(args: argparse.Namespace) -> None:
    ensure_output_dirs()
    set_seed(args.seed)
    device = torch.device("cpu")
    dataloader = build_dataloader(args.batch_size)

    generator = Generator(latent_dim=args.latent_dim).to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(16, args.latent_dim, 1, 1, device=device)
    best_score = float("-inf")

    for epoch in range(1, args.epochs + 1):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        running_loss_g = 0.0
        running_loss_d = 0.0
        processed_batches = 0
        for real_images, _ in loop:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            real_labels = torch.full((batch_size, 1), args.real_label_value, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            optimizer_d.zero_grad(set_to_none=True)
            real_preds = discriminator(real_images)
            loss_real = criterion(real_preds, real_labels)

            noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
            fake_images = generator(noise)
            fake_preds = discriminator(fake_images.detach())
            loss_fake = criterion(fake_preds, fake_labels)
            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            gen_preds = discriminator(fake_images)
            loss_g = criterion(gen_preds, real_labels)
            loss_g.backward()
            optimizer_g.step()

            running_loss_d += loss_d.item()
            running_loss_g += loss_g.item()
            processed_batches += 1
            loop.set_postfix(loss_d=f"{loss_d.item():.4f}", loss_g=f"{loss_g.item():.4f}")

            if args.max_batches and (loop.n >= args.max_batches):
                break

        metrics = evaluate_generator(generator, args.latent_dim, device, batch_size=32)
        checkpoint_payload = {
            "epoch": epoch,
            "latent_dim": args.latent_dim,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "optimizer_d_state_dict": optimizer_d.state_dict(),
            "metrics": metrics,
            "config": vars(args),
        }
        save_checkpoint(CHECKPOINTS_DIR / "last.pt", checkpoint_payload)

        if metrics["quality_score"] > best_score:
            best_score = metrics["quality_score"]
            save_checkpoint(CHECKPOINTS_DIR / "best.pt", checkpoint_payload)

        if epoch % args.sample_every == 0:
            with torch.no_grad():
                samples = generator(fixed_noise)
            save_image_grid(
                samples,
                SAMPLES_DIR / f"train_epoch_{epoch:03d}.png",
                title=f"Epoch {epoch} | score={metrics['quality_score']:.4f}",
            )

        print(
            f"epoch={epoch} loss_d={running_loss_d / processed_batches:.4f} "
            f"loss_g={running_loss_g / processed_batches:.4f} "
            f"feature_distance={metrics['feature_distance']:.4f} quality_score={metrics['quality_score']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small DCGAN on Fashion-MNIST.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--sample_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real_label_value", type=float, default=0.9, help="One-sided label smoothing value.")
    parser.add_argument("--max_batches", type=int, default=0, help="Optional cap for quick CPU smoke tests.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
