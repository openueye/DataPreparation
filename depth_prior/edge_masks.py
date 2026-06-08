from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


def summarize(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()) if array.size else 0.0,
        "median": float(np.median(array)) if array.size else 0.0,
        "min": float(array.min()) if array.size else 0.0,
        "max": float(array.max()) if array.size else 0.0,
    }


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


def edge_strength_inverse_depth(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    inverse_depth = np.zeros_like(depth, dtype=np.float32)
    inverse_depth[valid] = 1.0 / np.maximum(depth[valid], 1e-6)
    grad_x = cv2.Sobel(inverse_depth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(inverse_depth, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    edge[~valid] = 0.0
    return edge


def _label_from_percentile(percentile: float) -> str:
    return f"edge{int(percentile):02d}"


def generate_edge_masks(args: argparse.Namespace) -> Dict[str, Dict[str, object]]:
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    completed_dir = (
        Path(args.completed_depths_dir)
        if args.completed_depths_dir is not None
        else scene_dir / "depths_spnet_fused9_anchor_preserved"
    ).expanduser().resolve()
    anchor_dir = (
        Path(args.anchor_depths_dir)
        if args.anchor_depths_dir is not None
        else scene_dir / "depths_lidar_fused_9f"
    ).expanduser().resolve()
    stems = sorted(path.stem for path in completed_dir.glob("*.npy"))
    if not stems:
        raise FileNotFoundError(f"No .npy depth maps found under {completed_dir}")

    reports: Dict[str, Dict[str, object]] = {}
    for percentile in args.percentiles:
        label = _label_from_percentile(percentile)
        mask_dir = scene_dir / f"masks_spnet_fused9_anchor_preserved_{label}"
        report_path = scene_dir / f"depth_prior_report_spnet_fused9_anchor_preserved_{label}_mask.json"
        if mask_dir.exists() and any(mask_dir.glob("*.npy")) and not args.overwrite:
            raise FileExistsError(f"{mask_dir} already contains masks; pass --overwrite to replace.")
        mask_dir.mkdir(parents=True, exist_ok=True)

        ratios: List[float] = []
        anchor_ratios: List[float] = []
        thresholds: List[float] = []
        per_image = []
        for stem in stems:
            depth = load_depth(completed_dir / f"{stem}.npy")
            anchor_depth = load_depth(anchor_dir / f"{stem}.npy")
            if depth.shape != anchor_depth.shape:
                raise ValueError(f"Shape mismatch for {stem}: depth={depth.shape} anchor={anchor_depth.shape}")

            valid = np.isfinite(depth) & (depth >= args.target_min) & (depth <= args.target_max)
            anchor = np.isfinite(anchor_depth) & (anchor_depth > 0.0)
            candidate = valid & ~anchor
            edge = edge_strength_inverse_depth(depth, valid)
            if candidate.any():
                threshold = float(np.percentile(edge[candidate], percentile))
                gated = valid & ((edge <= threshold) | anchor)
            else:
                threshold = 0.0
                gated = valid

            mask_path = mask_dir / f"{stem}.npy"
            np.save(mask_path, gated.astype(np.uint8, copy=False))
            ratio = float(gated.mean())
            anchor_ratio = float(anchor.mean())
            ratios.append(ratio)
            anchor_ratios.append(anchor_ratio)
            thresholds.append(threshold)
            per_image.append(
                {
                    "image": f"{stem}.jpg",
                    "mask_path": str(mask_path),
                    "valid_ratio": ratio,
                    "anchor_ratio": anchor_ratio,
                    "completed_valid_ratio": float(valid.mean()),
                    "edge_threshold": threshold,
                    "kept_non_anchor_ratio": float((gated & ~anchor).sum() / max((~anchor).sum(), 1)),
                }
            )

        report = {
            "scene_dir": str(scene_dir),
            "method": f"spnet-fused9-anchor-preserved-{label}-mask",
            "completed_depths_dir": str(completed_dir),
            "anchor_depths_dir": str(anchor_dir),
            "masks_dir": str(mask_dir),
            "edge_metric": "Sobel magnitude on inverse depth",
            "edge_percentile_keep": float(percentile),
            "anchor_preserve": True,
            "target_depth_range": {"min": float(args.target_min), "max": float(args.target_max)},
            "num_images": len(per_image),
            "valid_ratio_summary": summarize(ratios),
            "anchor_ratio_summary": summarize(anchor_ratios),
            "edge_threshold_summary": summarize(thresholds),
            "per_image": per_image,
            "contract": {
                "mask_semantics": "uint8 mask, 1 means completed depth may be used; anchor pixels are always kept.",
                "depths_dir": str(completed_dir),
            },
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        reports[label] = report
    return reports


def apply_depth_mask(args: argparse.Namespace) -> Dict[str, object]:
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    input_depths_dir = (
        Path(args.input_depths_dir)
        if args.input_depths_dir is not None
        else scene_dir / "depths_spnet_fused9_anchor_preserved"
    ).expanduser().resolve()
    input_masks_dir = (
        Path(args.input_masks_dir)
        if args.input_masks_dir is not None
        else scene_dir / "masks_spnet_fused9_anchor_preserved_edge90"
    ).expanduser().resolve()
    anchor_depths_dir = (
        Path(args.anchor_depths_dir)
        if args.anchor_depths_dir is not None
        else scene_dir / "depths_lidar_fused_9f"
    ).expanduser().resolve()
    output_depths_dir = (
        Path(args.output_depths_dir)
        if args.output_depths_dir is not None
        else scene_dir / "depths_spnet_fused9_anchor_preserved_edge90"
    ).expanduser().resolve()
    output_report = (
        Path(args.output_report)
        if args.output_report is not None
        else scene_dir / "depth_prior_report_spnet_fused9_anchor_preserved_edge90.json"
    ).expanduser().resolve()

    if output_depths_dir.exists() and any(output_depths_dir.glob("*.npy")) and not args.overwrite:
        raise FileExistsError(f"{output_depths_dir} already contains .npy files; pass --overwrite.")
    output_depths_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted(path.stem for path in input_depths_dir.glob("*.npy"))
    if not stems:
        raise FileNotFoundError(f"No .npy depth maps found under {input_depths_dir}")

    ratios: List[float] = []
    anchor_ratios: List[float] = []
    anchor_max_errors: List[float] = []
    per_image = []
    for stem in stems:
        depth = load_depth(input_depths_dir / f"{stem}.npy")
        mask = np.asarray(np.load(input_masks_dir / f"{stem}.npy")).astype(bool)
        anchor = load_depth(anchor_depths_dir / f"{stem}.npy")
        if depth.shape != mask.shape or depth.shape != anchor.shape:
            raise ValueError(f"Shape mismatch for {stem}: depth={depth.shape} mask={mask.shape} anchor={anchor.shape}")

        output = depth.copy()
        output[~mask] = 0.0
        anchor_mask = np.isfinite(anchor) & (anchor > 0.0)
        anchor_error = float(np.abs(output[anchor_mask] - anchor[anchor_mask]).max()) if anchor_mask.any() else 0.0
        output_path = output_depths_dir / f"{stem}.npy"
        np.save(output_path, output.astype(np.float32, copy=False))

        valid = np.isfinite(output) & (output > 0.0)
        ratio = float(valid.mean())
        anchor_ratio = float(anchor_mask.mean())
        ratios.append(ratio)
        anchor_ratios.append(anchor_ratio)
        anchor_max_errors.append(anchor_error)
        per_image.append(
            {
                "image": f"{stem}.jpg",
                "depth_path": str(output_path),
                "valid_ratio": ratio,
                "anchor_ratio": anchor_ratio,
                "anchor_max_abs_error": anchor_error,
            }
        )

    report = {
        "scene_dir": str(scene_dir),
        "method": args.method,
        "source_depths_dir": str(input_depths_dir),
        "source_masks_dir": str(input_masks_dir),
        "anchor_depths_dir": str(anchor_depths_dir),
        "depths_dir": str(output_depths_dir),
        "depth_convention": "metric",
        "unit": "meter",
        "anchor_preserve": True,
        "num_images": len(per_image),
        "valid_ratio_summary": summarize(ratios),
        "anchor_ratio_summary": summarize(anchor_ratios),
        "anchor_max_abs_error_summary": summarize(anchor_max_errors),
        "per_image": per_image,
        "contract": {
            "supervision_geometry": "Completed metric depth with explicit mask applied by zeroing invalid pixels; anchor pixels are expected to be preserved by the mask.",
            "shape": "matches original RGB image shape",
            "valid_mask": "finite depth > 0 after masking",
        },
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_edge_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create edge-gated masks for SPNet completed depth priors.")
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--completed-depths-dir", type=Path, default=None)
    parser.add_argument("--anchor-depths-dir", type=Path, default=None)
    parser.add_argument("--percentiles", type=float, nargs="+", default=[90.0])
    parser.add_argument("--target-min", type=float, default=1.0)
    parser.add_argument("--target-max", type=float, default=50.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def parse_apply_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply uint8 depth prior masks by zeroing invalid pixels.")
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--input-depths-dir", type=Path, default=None)
    parser.add_argument("--input-masks-dir", type=Path, default=None)
    parser.add_argument("--anchor-depths-dir", type=Path, default=None)
    parser.add_argument("--output-depths-dir", type=Path, default=None)
    parser.add_argument("--output-report", type=Path, default=None)
    parser.add_argument("--method", default="spnet-fused9-anchor-preserved-edge90")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    command = Path(sys.argv[0]).name
    if command == "depth-prior-apply-mask":
        report = apply_depth_mask(parse_apply_args(argv))
        print(f"[INFO] wrote {report['depths_dir']}")
        print(f"[INFO] mean_valid_ratio={report['valid_ratio_summary']['mean']:.6f}")
        return 0

    reports = generate_edge_masks(parse_edge_args(argv))
    for label, report in reports.items():
        print(f"[INFO] {label} mean_valid_ratio={report['valid_ratio_summary']['mean']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
