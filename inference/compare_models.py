from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs" / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.generate import load_generator
from utils.io import COMPARISONS_DIR, CHECKPOINTS_DIR, ensure_output_dirs, resolve_quality_checkpoint
from utils.metrics import save_image_grid
from utils.seed import set_seed


def discover_models() -> List[Tuple[str, Path]]:
    candidates = [
        ("baseline", resolve_quality_checkpoint()),
        ("pruned_20", CHECKPOINTS_DIR / "pruned_20.pt"),
        ("pruned_40", CHECKPOINTS_DIR / "pruned_40.pt"),
        ("pruned_60", CHECKPOINTS_DIR / "pruned_60.pt"),
        ("quantized", CHECKPOINTS_DIR / "quantized_int8.pt"),
    ]
    return [(name, path) for name, path in candidates if path.exists()]


def shared_latents(latent_dim: int, num_samples: int, seed: int, path: Path) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn(num_samples, latent_dim, 1, 1, generator=generator)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"seed": seed, "latent_dim": latent_dim, "noise": noise}, path)
    return noise


def visual_observation(model_name: str) -> str:
    if model_name == "baseline":
        return "Quality reference model with the fullest detail retained."
    if model_name == "pruned_20":
        return "Light pruning; visuals should stay close to baseline with mild softening."
    if model_name == "pruned_40":
        return "Moderate pruning; expect cleaner speed-size tradeoff with some loss of sharpness."
    if model_name == "pruned_60":
        return "Aggressive pruning; inspect for softer boundaries and reduced garment structure."
    if model_name == "quantized":
        return "Weight-only int8 artifact; inspect for contrast shifts after dequantized runtime loading."
    return "Prototype comparison artifact."


def combined_figure(grids: List[Tuple[str, torch.Tensor]], output_path: Path) -> None:
    rows = len(grids)
    fig, axes = plt.subplots(rows, 1, figsize=(8, 2.6 * rows))
    if rows == 1:
        axes = [axes]
    for axis, (label, images) in zip(axes, grids):
        grid = make_grid(images.detach().cpu(), nrow=4, normalize=True, value_range=(-1, 1))
        axis.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
        axis.set_title(label, fontsize=12)
        axis.axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    ensure_output_dirs()
    set_seed(args.seed)

    models = discover_models()
    if not models:
        raise FileNotFoundError("No comparison models found. Run training/compression first.")

    baseline_generator, _ = load_generator(models[0][1])
    latent_dim = getattr(baseline_generator, "latent_dim", args.latent_dim)
    latent_path = COMPARISONS_DIR / "shared_latents.pt"
    noise = shared_latents(latent_dim, args.num_samples, args.seed, latent_path)

    rows = []
    combined_rows: List[Tuple[str, torch.Tensor]] = []
    for model_name, checkpoint_path in models:
        generator, load_mode = load_generator(checkpoint_path)
        generator.eval()
        with torch.no_grad():
            images = generator(noise)
        grid_path = COMPARISONS_DIR / f"{model_name}_grid.png"
        label = f"{model_name} ({load_mode})"
        save_image_grid(images, grid_path, title=label, nrow=args.nrow)
        combined_rows.append((label, images))
        rows.append(
            "\n".join(
                [
                    f"## {model_name}",
                    "",
                    f"- artifact: `{checkpoint_path}`",
                    f"- load mode: `{load_mode}`",
                    f"- shared latents: `{latent_path.name}`",
                    f"- image: `{grid_path.name}`",
                    f"- note: {visual_observation(model_name)}",
                    "",
                ]
            )
        )

    combined_path = COMPARISONS_DIR / "combined_comparison.png"
    combined_figure(combined_rows, combined_path)

    summary_lines = [
        "# EdgeGen Model Comparison",
        "",
        "All model variants below were generated from the same saved latent vectors for a fair visual comparison.",
        "",
        f"- shared latent file: `{latent_path}`",
        f"- combined comparison: `{combined_path}`",
        "",
    ]
    summary_lines.extend(rows)
    summary_path = COMPARISONS_DIR / "comparison_summary.md"
    summary_path.write_text("\n".join(summary_lines))

    print(f"Saved comparison outputs to {COMPARISONS_DIR}")
    print(f"Combined comparison image: {combined_path}")
    print(f"Comparison markdown: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fair side-by-side sample comparisons across model variants.")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
