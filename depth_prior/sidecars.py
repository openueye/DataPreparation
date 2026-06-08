from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def positive_finite(array: np.ndarray) -> np.ndarray:
    return np.isfinite(array) & (array > 0.0)


def load_depth(path: Path) -> np.ndarray:
    depth = np.asarray(np.load(path), dtype=np.float32)
    if depth.ndim == 3:
        if depth.shape[0] == 1:
            depth = depth[0]
        elif depth.shape[-1] == 1:
            depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected HxW depth map, got {depth.shape}: {path}")
    return depth


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def generate_sidecars(args: argparse.Namespace) -> Dict[str, object]:
    final_depths_dir = Path(args.final_depths_dir).expanduser().resolve()
    completed_depths_dir = Path(args.completed_depths_dir).expanduser().resolve()
    anchor_depths_dir = Path(args.anchor_depths_dir).expanduser().resolve()
    confidence_dir = Path(args.confidence_dir).expanduser().resolve()
    source_labels_dir = Path(args.source_labels_dir).expanduser().resolve()
    confidence_dir.mkdir(parents=True, exist_ok=True)
    source_labels_dir.mkdir(parents=True, exist_ok=True)

    final_paths = sorted(final_depths_dir.glob("*.npy"))
    if not final_paths:
        raise FileNotFoundError(f"No .npy depth maps found under {final_depths_dir}")

    per_image = []
    anchor_ratios: List[float] = []
    completed_ratios: List[float] = []
    rejected_ratios: List[float] = []
    mean_weights: List[float] = []

    for final_path in final_paths:
        stem = final_path.stem
        completed_path = completed_depths_dir / final_path.name
        anchor_path = anchor_depths_dir / final_path.name
        if not completed_path.exists():
            raise FileNotFoundError(f"Missing completed depth map for {stem}: {completed_path}")
        if not anchor_path.exists():
            raise FileNotFoundError(f"Missing anchor depth map for {stem}: {anchor_path}")

        final_depth = load_depth(final_path)
        completed_depth = load_depth(completed_path)
        anchor_depth = load_depth(anchor_path)
        if final_depth.shape != completed_depth.shape or final_depth.shape != anchor_depth.shape:
            raise ValueError(
                f"Shape mismatch for {stem}: final={final_depth.shape}, "
                f"completed={completed_depth.shape}, anchor={anchor_depth.shape}"
            )

        final_valid = positive_finite(final_depth)
        completed_valid = positive_finite(completed_depth)
        anchor_valid = positive_finite(anchor_depth)
        anchor_kept = final_valid & anchor_valid
        completed_kept = final_valid & ~anchor_valid
        rejected = completed_valid & ~final_valid

        confidence = np.zeros(final_depth.shape, dtype=np.float32)
        confidence[anchor_kept] = float(args.anchor_weight)
        confidence[completed_kept] = float(args.completion_weight)

        source_labels = np.zeros(final_depth.shape, dtype=np.uint8)
        source_labels[rejected] = 4
        source_labels[completed_kept] = 2
        source_labels[anchor_kept] = 1

        confidence_path = confidence_dir / final_path.name
        labels_path = source_labels_dir / final_path.name
        np.save(confidence_path, confidence)
        np.save(labels_path, source_labels)

        pixel_count = int(final_depth.size)
        positive_weight = confidence[confidence > 0.0]
        anchor_ratio = float(anchor_kept.sum() / pixel_count)
        completed_ratio = float(completed_kept.sum() / pixel_count)
        rejected_ratio = float(rejected.sum() / pixel_count)
        mean_weight = float(positive_weight.mean()) if positive_weight.size else 0.0
        anchor_ratios.append(anchor_ratio)
        completed_ratios.append(completed_ratio)
        rejected_ratios.append(rejected_ratio)
        mean_weights.append(mean_weight)
        per_image.append(
            {
                "image_stem": stem,
                "confidence_path": str(confidence_path),
                "source_labels_path": str(labels_path),
                "anchor_ratio": anchor_ratio,
                "completion_ratio": completed_ratio,
                "rejected_ratio": rejected_ratio,
                "mean_positive_weight": mean_weight,
                "anchor_pixels": int(anchor_kept.sum()),
                "completion_pixels": int(completed_kept.sum()),
                "rejected_pixels": int(rejected.sum()),
                "anchor_valid_but_final_invalid_pixels": int((anchor_valid & ~final_valid).sum()),
            }
        )

    report = {
        "method": args.method,
        "final_depths_dir": str(final_depths_dir),
        "completed_depths_dir": str(completed_depths_dir),
        "anchor_depths_dir": str(anchor_depths_dir),
        "confidence_dir": str(confidence_dir),
        "source_labels_dir": str(source_labels_dir),
        "anchor_weight": float(args.anchor_weight),
        "completion_weight": float(args.completion_weight),
        "invalid_or_rejected_weight": 0.0,
        "label_contract": {
            "0": "invalid_or_never_completed",
            "1": "lidar_anchor",
            "2": "spnet_completed_kept",
            "4": "completed_rejected_by_final_mask",
        },
        "num_images": len(final_paths),
        "anchor_ratio_summary": summarize(anchor_ratios),
        "completion_ratio_summary": summarize(completed_ratios),
        "rejected_ratio_summary": summarize(rejected_ratios),
        "mean_positive_weight_summary": summarize(mean_weights),
        "per_image": per_image,
    }
    if args.report_path:
        report_path = Path(args.report_path).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate depth confidence and source-label sidecars.")
    parser.add_argument("--final-depths-dir", required=True, type=Path)
    parser.add_argument("--completed-depths-dir", required=True, type=Path)
    parser.add_argument("--anchor-depths-dir", required=True, type=Path)
    parser.add_argument("--confidence-dir", required=True, type=Path)
    parser.add_argument("--source-labels-dir", required=True, type=Path)
    parser.add_argument("--report-path", default="", type=Path)
    parser.add_argument("--method", default="spnet-fused9-anchor-preserved-edge90-confidence")
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--completion-weight", type=float, default=0.2)
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    report = generate_sidecars(parse_args(argv))
    print(json.dumps({k: report[k] for k in ("method", "num_images", "confidence_dir", "source_labels_dir")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
