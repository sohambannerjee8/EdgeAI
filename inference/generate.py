from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.dcgan import Generator
from utils.io import (
    CHECKPOINTS_DIR,
    SAMPLES_DIR,
    ensure_output_dirs,
    load_checkpoint,
    load_quantized_artifact,
    resolve_quality_checkpoint,
)
from utils.metrics import save_image_grid
from utils.seed import set_seed


def load_generator(path: Path) -> tuple[Generator, str]:
    if "quantized" in path.stem:
        artifact = load_quantized_artifact(path)
        generator = Generator(latent_dim=artifact["metadata"]["latent_dim"])
        generator.load_state_dict(artifact["state_dict"])
        return generator, artifact["metadata"]["method"]

    checkpoint = load_checkpoint(path)
    latent_dim = checkpoint.get("latent_dim", checkpoint.get("config", {}).get("latent_dim", 64))
    generator = Generator(latent_dim=latent_dim)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    return generator, "dense"


def main(args: argparse.Namespace) -> None:
    ensure_output_dirs()
    set_seed(args.seed)
    generator, load_mode = load_generator(args.checkpoint)
    generator.eval()
    device = torch.device("cpu")
    generator.to(device)

    with torch.no_grad():
        images = generator.sample(args.num_samples, device)

    output_path = SAMPLES_DIR / args.output_name
    save_image_grid(images, output_path, title=f"Generated samples ({load_mode})", nrow=args.nrow)
    print(f"Saved generated grid to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate samples from a trained generator checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=resolve_quality_checkpoint())
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--output_name", type=str, default="generated_grid.png")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
