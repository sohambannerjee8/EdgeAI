from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.generate import load_generator
from utils.io import SAMPLES_DIR, ensure_output_dirs, resolve_fast_checkpoint, resolve_quality_checkpoint
from utils.metrics import save_image_grid
from utils.seed import set_seed


def benchmark_single_batch(generator: torch.nn.Module, batch_size: int, latent_dim: int) -> float:
    device = torch.device("cpu")
    generator.eval()
    generator.to(device)
    noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
    start = time.perf_counter()
    with torch.no_grad():
        _ = generator(noise)
    end = time.perf_counter()
    return (end - start) * 1000.0


def main(args: argparse.Namespace) -> None:
    ensure_output_dirs()
    set_seed(args.seed)
    quality_path = args.quality_checkpoint
    fast_path = resolve_fast_checkpoint()
    selected_mode = "fast" if args.latency_budget_ms < args.threshold_ms else "quality"
    selected_path = fast_path if selected_mode == "fast" else quality_path

    generator, load_mode = load_generator(selected_path)
    latent_dim = getattr(generator, "latent_dim", args.latent_dim)
    latency_ms = benchmark_single_batch(generator, args.num_samples, latent_dim)

    with torch.no_grad():
        images = generator.sample(args.num_samples, "cpu")

    output_path = SAMPLES_DIR / f"adaptive_{selected_mode}.png"
    save_image_grid(images, output_path, title=f"Adaptive mode: {selected_mode}", nrow=args.nrow)

    print(f"Selected mode: {selected_mode}")
    print(f"Latency budget (ms): {args.latency_budget_ms:.2f}")
    print(f"Threshold (ms): {args.threshold_ms:.2f}")
    print(f"Checkpoint: {selected_path}")
    print(f"Load mode: {load_mode}")
    print(f"Observed single-batch latency (ms): {latency_ms:.2f}")
    print(f"Saved samples to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive inference controller for EdgeGen.")
    parser.add_argument("--latency_budget_ms", type=float, required=True)
    parser.add_argument("--threshold_ms", type=float, default=25.0)
    parser.add_argument("--quality_checkpoint", type=Path, default=resolve_quality_checkpoint())
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
