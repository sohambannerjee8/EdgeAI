from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.utils.prune as prune
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.dcgan import Generator
from utils.io import CHECKPOINTS_DIR, ensure_output_dirs, load_checkpoint, save_checkpoint
from utils.metrics import prototype_quality_score


def load_real_batch(batch_size: int = 32) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.FashionMNIST(root=PROJECT_ROOT / "data", train=False, download=True, transform=transform)
    return torch.stack([dataset[index][0] for index in range(batch_size)])


def prune_generator(generator: Generator, amount: float) -> tuple[Generator, dict[str, float], float]:
    modules_to_prune = []
    for module in generator.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
            modules_to_prune.append((module, "weight"))
    prune.global_unstructured(modules_to_prune, pruning_method=prune.L1Unstructured, amount=amount)

    layer_sparsities: dict[str, float] = {}
    total_zeros = 0
    total_params = 0
    for name, module in generator.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
            weight = module.weight
            zeros = torch.sum(weight == 0).item()
            params = weight.numel()
            layer_sparsities[name] = zeros / params
            total_zeros += zeros
            total_params += params
            prune.remove(module, "weight")
    overall = total_zeros / total_params
    return generator, layer_sparsities, overall


def main(args: argparse.Namespace) -> None:
    ensure_output_dirs()
    baseline_ckpt = load_checkpoint(args.checkpoint)
    real_batch = load_real_batch(batch_size=args.eval_batch_size)
    device = torch.device("cpu")
    best_ratio = None
    best_score = float("-inf")

    for ratio in args.ratios:
        generator = Generator(latent_dim=baseline_ckpt["latent_dim"]).to(device)
        generator.load_state_dict(baseline_ckpt["generator_state_dict"])
        generator.eval()
        generator, layer_sparsities, overall_sparsity = prune_generator(generator, amount=ratio)

        with torch.no_grad():
            fake_batch = generator.sample(args.eval_batch_size, device)
        metrics = prototype_quality_score(real_batch.to(device), fake_batch)

        tag = int(ratio * 100)
        output_path = CHECKPOINTS_DIR / f"pruned_{tag}.pt"
        save_checkpoint(
            output_path,
            {
                "latent_dim": baseline_ckpt["latent_dim"],
                "generator_state_dict": generator.state_dict(),
                "prune_ratio": ratio,
                "layer_sparsities": layer_sparsities,
                "overall_sparsity": overall_sparsity,
                "metrics": metrics,
                "source_checkpoint": str(args.checkpoint),
            },
        )

        print(f"\nPruning ratio: {ratio:.1f}")
        for layer_name, sparsity in layer_sparsities.items():
            print(f"  {layer_name}: sparsity={sparsity:.4f}")
        print(f"  overall_sparsity={overall_sparsity:.4f}")
        print(f"  feature_distance={metrics['feature_distance']:.4f} quality_score={metrics['quality_score']:.4f}")

        if metrics["quality_score"] > best_score:
            best_score = metrics["quality_score"]
            best_ratio = ratio

    if best_ratio is not None:
        best_tag = int(best_ratio * 100)
        source = CHECKPOINTS_DIR / f"pruned_{best_tag}.pt"
        target = CHECKPOINTS_DIR / "best_pruned.pt"
        shutil.copyfile(source, target)
        print(f"\nSaved best pruned checkpoint: {target} (ratio={best_ratio:.1f})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply magnitude pruning to the generator.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINTS_DIR / "best.pt")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.2, 0.4, 0.6])
    parser.add_argument("--eval_batch_size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
