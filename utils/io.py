from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_ROOT / "checkpoints"
SAMPLES_DIR = OUTPUTS_ROOT / "samples"
COMPARISONS_DIR = SAMPLES_DIR / "comparisons"
BENCHMARKS_DIR = OUTPUTS_ROOT / "benchmarks"


def ensure_output_dirs() -> None:
    for directory in [CHECKPOINTS_DIR, SAMPLES_DIR, COMPARISONS_DIR, BENCHMARKS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def resolve_quality_checkpoint() -> Path:
    candidates = [
        CHECKPOINTS_DIR / "best.pt",
        CHECKPOINTS_DIR / "last.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return CHECKPOINTS_DIR / "best.pt"


def resolve_fast_checkpoint() -> Path:
    candidates = [
        CHECKPOINTS_DIR / "quantized_int8.pt",
        CHECKPOINTS_DIR / "best_pruned.pt",
        CHECKPOINTS_DIR / "pruned_60.pt",
        CHECKPOINTS_DIR / "pruned_40.pt",
        CHECKPOINTS_DIR / "pruned_20.pt",
        resolve_quality_checkpoint(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return resolve_quality_checkpoint()


def save_checkpoint(path: Union[str, Path], payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: Union[str, Path], map_location: Union[str, torch.device] = "cpu"
) -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


def checkpoint_size_mb(path: Union[str, Path]) -> float:
    return Path(path).stat().st_size / (1024 * 1024)


def quantize_tensor_int8(tensor: torch.Tensor) -> dict[str, Any]:
    if not tensor.is_floating_point():
        return {"dtype": str(tensor.dtype), "tensor": tensor.cpu()}
    max_abs = tensor.abs().max().item()
    scale = max(max_abs / 127.0, 1e-8)
    q_tensor = torch.clamp((tensor / scale).round(), -127, 127).to(torch.int8).cpu()
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "scale": scale,
        "tensor": q_tensor,
    }


def dequantize_tensor_int8(payload: dict[str, Any]) -> torch.Tensor:
    if "scale" not in payload:
        return payload["tensor"]
    tensor = payload["tensor"].float() * payload["scale"]
    return tensor.view(payload["shape"])


def save_quantized_artifact(
    path: Union[str, Path], state_dict: dict[str, torch.Tensor], metadata: dict[str, Any]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    quantized = {name: quantize_tensor_int8(tensor) for name, tensor in state_dict.items()}
    torch.save({"metadata": metadata, "state_dict": quantized}, path)


def load_quantized_artifact(
    path: Union[str, Path], map_location: Union[str, torch.device] = "cpu"
) -> dict[str, Any]:
    artifact = torch.load(Path(path), map_location=map_location)
    artifact["state_dict"] = {
        name: dequantize_tensor_int8(payload) for name, payload in artifact["state_dict"].items()
    }
    return artifact


def save_json(path: Union[str, Path], payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
