from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.dcgan import Generator
from utils.io import CHECKPOINTS_DIR, checkpoint_size_mb, ensure_output_dirs, load_checkpoint, save_quantized_artifact


def main(args: argparse.Namespace) -> None:
    ensure_output_dirs()
    checkpoint = load_checkpoint(args.checkpoint)
    generator = Generator(latent_dim=checkpoint["latent_dim"])
    generator.load_state_dict(checkpoint["generator_state_dict"])

    output_path = CHECKPOINTS_DIR / "quantized_int8.pt"
    metadata = {
        "method": "weight_only_int8",
        "runtime_note": (
            "Generator contains ConvTranspose layers, so broad CPU int8 execution is not reliably supported. "
            "This artifact stores weights as int8 with per-tensor scales and dequantizes to float32 on load."
        ),
        "latent_dim": checkpoint["latent_dim"],
        "source_checkpoint": str(args.checkpoint),
    }
    save_quantized_artifact(output_path, generator.state_dict(), metadata)

    baseline_size = checkpoint_size_mb(args.checkpoint)
    quantized_size = checkpoint_size_mb(output_path)
    reduction = 100.0 * (baseline_size - quantized_size) / max(baseline_size, 1e-8)

    print("Quantization summary")
    print(f"  baseline_checkpoint={args.checkpoint}")
    print(f"  baseline_size_mb={baseline_size:.4f}")
    print(f"  quantized_artifact={output_path}")
    print(f"  quantized_size_mb={quantized_size:.4f}")
    print(f"  size_reduction_percent={reduction:.2f}")
    print(f"  note={metadata['runtime_note']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save a weight-only int8 quantized generator artifact.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINTS_DIR / "best.pt")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
