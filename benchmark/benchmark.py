from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import psutil
import torch
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.generate import load_generator
from utils.io import BENCHMARKS_DIR, CHECKPOINTS_DIR, checkpoint_size_mb, ensure_output_dirs, resolve_quality_checkpoint
from utils.metrics import format_quality_note, prototype_quality_score, save_image_grid


def load_real_batch(batch_size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.FashionMNIST(root=PROJECT_ROOT / "data", train=False, download=True, transform=transform)
    return torch.stack([dataset[index][0] for index in range(batch_size)])


def benchmark_generator(generator: torch.nn.Module, batch_size: int, warmup_iters: int, measure_iters: int) -> tuple[float, float, float]:
    device = torch.device("cpu")
    generator.eval()
    generator.to(device)
    latent_dim = getattr(generator, "latent_dim", 64)
    noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)

    for _ in range(warmup_iters):
        with torch.no_grad():
            _ = generator(noise)

    process = psutil.Process()
    rss_before = process.memory_info().rss
    latencies = []
    peak_rss = rss_before
    for _ in range(measure_iters):
        start = time.perf_counter()
        with torch.no_grad():
            _ = generator(noise)
        latency = time.perf_counter() - start
        latencies.append(latency)
        peak_rss = max(peak_rss, process.memory_info().rss)

    avg_latency_ms = 1000.0 * sum(latencies) / len(latencies)
    throughput = batch_size / (sum(latencies) / len(latencies))
    peak_memory_mb = (peak_rss - rss_before) / (1024 * 1024)
    return avg_latency_ms, throughput, peak_memory_mb


def summary_markdown(df: pd.DataFrame, best_pruned_name: str) -> str:
    header = "# EdgeGen Benchmark Summary\n\n"
    note = f"{format_quality_note()}\n\n"
    best_row = df.loc[df["model_name"] == best_pruned_name].iloc[0]
    lines = [
        f"- Best pruned model by proxy quality: `{best_pruned_name}`",
        f"- Best pruned quality score: `{best_row['quality_score']:.4f}`",
        f"- Best pruned feature distance: `{best_row['feature_distance']:.4f}`",
        "",
        df.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- Lower latency and higher throughput indicate better CPU deployment efficiency.",
        "- Lower feature distance and higher quality score indicate better prototype-level sample fidelity.",
        "- Quantized runtime uses dequantized float32 execution when ConvTranspose kernels limit native int8 inference.",
    ]
    return header + note + "\n".join(lines) + "\n"


def visual_quality_note(model_name: str, quality_score: float, best_score: float, min_score: float) -> str:
    if model_name == "baseline":
        return "Baseline quality model; use this as the primary visual reference."
    if model_name == "best_pruned":
        return "Selected compressed quality checkpoint; best overall pruning trade-off for presentation."
    if model_name == "quantized":
        return "Most compact artifact; inspect for mild contrast or texture loss after dequantized loading."
    if quality_score >= best_score - 1e-6:
        return "Strongest compressed visual result in this benchmark run."
    if quality_score <= min_score + 1e-6:
        return "Weakest visual score in this run; likely softer outlines and less stable garments."
    if "60" in model_name:
        return "Aggressive compression with noticeable detail risk but strong edge efficiency."
    if "40" in model_name:
        return "Moderate compression; balanced output with some softening."
    if "20" in model_name:
        return "Light compression; typically closest visual match to baseline."
    return "Prototype-level visual observation."


def professor_summary_markdown(df: pd.DataFrame) -> str:
    lines = [
        "# EdgeGen Professor Summary",
        "",
        "This report condenses the CPU deployment comparison for presentation use.",
        "",
        "| Model | Size (MB) | Latency (ms) | Throughput (samples/s) | Peak Memory (MB) | Visual Observation |",
        "|:--|--:|--:|--:|--:|:--|",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| {model_name} | {model_size_mb:.3f} | {avg_latency_ms:.3f} | {throughput_samples_per_sec:.2f} | "
            "{peak_memory_mb:.3f} | {visual_quality_note} |".format(**row.to_dict())
        )
    lines.extend(
        [
            "",
            "Prototype note: visual observations are qualitative and should be interpreted alongside saved sample grids.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(args: argparse.Namespace) -> None:
    ensure_output_dirs()
    real_batch = load_real_batch(args.batch_size)

    model_paths: list[tuple[str, Path]] = [
        ("baseline", resolve_quality_checkpoint()),
        ("pruned_20", CHECKPOINTS_DIR / "pruned_20.pt"),
        ("pruned_40", CHECKPOINTS_DIR / "pruned_40.pt"),
        ("pruned_60", CHECKPOINTS_DIR / "pruned_60.pt"),
        ("best_pruned", CHECKPOINTS_DIR / "best_pruned.pt"),
    ]
    quantized_path = CHECKPOINTS_DIR / "quantized_int8.pt"
    if quantized_path.exists():
        model_paths.append(("quantized", quantized_path))

    rows = []
    best_pruned_name = "best_pruned"
    best_pruned_score = float("-inf")

    for name, path in model_paths:
        if not path.exists():
            print(f"Skipping missing model: {path}")
            continue
        generator, load_mode = load_generator(path)
        avg_latency_ms, throughput, peak_memory_mb = benchmark_generator(
            generator, args.batch_size, args.warmup_iters, args.measure_iters
        )
        with torch.no_grad():
            fake_batch = generator.sample(args.batch_size, "cpu")
        metrics = prototype_quality_score(real_batch, fake_batch)
        save_image_grid(fake_batch[:16], BENCHMARKS_DIR / f"{name}_samples.png", title=f"{name} ({load_mode})")

        row = {
            "model_name": name,
            "artifact_path": str(path),
            "load_mode": load_mode,
            "model_size_mb": checkpoint_size_mb(path),
            "avg_latency_ms": avg_latency_ms,
            "throughput_samples_per_sec": throughput,
            "peak_memory_mb": peak_memory_mb,
            "feature_distance": metrics["feature_distance"],
            "quality_score": metrics["quality_score"],
        }
        rows.append(row)

        if name.startswith("pruned") or name == "best_pruned":
            if metrics["quality_score"] > best_pruned_score:
                best_pruned_score = metrics["quality_score"]
                best_pruned_name = name

    df = pd.DataFrame(rows)
    max_quality = df["quality_score"].max()
    min_quality = df["quality_score"].min()
    df["visual_quality_note"] = df.apply(
        lambda row: visual_quality_note(row["model_name"], row["quality_score"], max_quality, min_quality),
        axis=1,
    )
    df = df.sort_values("avg_latency_ms").reset_index(drop=True)
    csv_path = BENCHMARKS_DIR / "results.csv"
    md_path = BENCHMARKS_DIR / "summary.md"
    professor_path = BENCHMARKS_DIR / "professor_summary.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(summary_markdown(df, best_pruned_name))
    professor_path.write_text(professor_summary_markdown(df))
    print(f"Saved benchmark CSV to {csv_path}")
    print(f"Saved benchmark summary to {md_path}")
    print(f"Saved professor summary to {professor_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark EdgeGen generator variants.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--measure_iters", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
