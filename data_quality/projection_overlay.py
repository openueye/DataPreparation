from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw

try:
    from data_preparation.shared.camera_models import load_camera_json, project_pinhole_k_like
    from data_preparation.shared.io import find_image_path, load_csv_rows_by_key, load_json, safe_float, write_json
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from data_preparation.shared.camera_models import load_camera_json, project_pinhole_k_like
    from data_preparation.shared.io import find_image_path, load_csv_rows_by_key, load_json, safe_float, write_json


DEFAULT_FRAMES = ["000000", "000500", "001000", "001500", "002000", "002500", "003000"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check LiDAR/RGB projection quality for a processed SLAM scene.")
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--frames", nargs="+", default=DEFAULT_FRAMES)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--max-overlay-points", type=int, default=12000)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--min-range", type=float, default=1e-6)
    parser.add_argument("--near-percentile", type=float, default=2.0)
    parser.add_argument("--far-percentile", type=float, default=98.0)
    parser.add_argument("--good-front-ratio", type=float, default=0.95)
    parser.add_argument("--good-coverage-ratio", type=float, default=0.35)
    parser.add_argument("--warn-front-ratio", type=float, default=0.80)
    parser.add_argument("--warn-coverage-ratio", type=float, default=0.10)
    parser.add_argument("--good-median-dt-ms", type=float, default=30.0)
    parser.add_argument("--warn-median-dt-ms", type=float, default=80.0)
    return parser.parse_args()


def load_associations(path: Path) -> Dict[str, dict]:
    return load_csv_rows_by_key(path, "frame_id")


def resolve_scene_paths(scene_dir: Path) -> Dict[str, Path]:
    paths = {
        "images": scene_dir / "images",
        "intrinsics": scene_dir / "intrinsics" / "camera.json",
        "tf_chain": scene_dir / "transforms" / "tf_chain.json",
        "associations": scene_dir / "metadata" / "associations.csv",
        "metadata": scene_dir / "metadata",
        "lidar_frames": scene_dir / "lidar" / "frames",
    }
    for key in ("images", "intrinsics", "tf_chain", "metadata", "lidar_frames"):
        if not paths[key].exists():
            raise FileNotFoundError(f"Missing required {key} path: {paths[key]}")
    return paths


def find_cloud_path(scene_dir: Path, lidar_frames_dir: Path, associations: Dict[str, dict], frame_id: str) -> Path:
    row = associations.get(frame_id)
    if row and row.get("cloud_path"):
        candidate = scene_dir / row["cloud_path"]
        if candidate.exists():
            return candidate
    candidate = lidar_frames_dir / f"{frame_id}.npy"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"No LiDAR frame found for frame {frame_id}")


def transform_lidar_to_camera(points_lidar: np.ndarray, camera_from_lidar: np.ndarray) -> np.ndarray:
    ones = np.ones((points_lidar.shape[0], 1), dtype=np.float64)
    points_h = np.concatenate([points_lidar.astype(np.float64), ones], axis=1)
    return (camera_from_lidar @ points_h.T).T[:, :3]


def depth_to_rgb(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    if far <= near:
        far = near + 1.0
    t = np.clip((depth - near) / (far - near), 0.0, 1.0)
    r = np.clip(1.5 - 2.0 * t, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * t - 1.0), 0.0, 1.0)
    b = np.clip(2.0 * t - 0.5, 0.0, 1.0)
    return (np.stack([r, g, b], axis=1) * 255.0).astype(np.uint8)


def draw_overlay(
    image_path: Path,
    output_path: Path,
    uv_inside: np.ndarray,
    depth_inside: np.ndarray,
    max_points: int,
    point_radius: int,
    near_percentile: float,
    far_percentile: float,
) -> int:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    if uv_inside.shape[0] == 0:
        image.save(output_path, quality=95)
        return 0

    draw_indices = np.arange(uv_inside.shape[0])
    if max_points > 0 and draw_indices.size > max_points:
        rng = np.random.default_rng(0)
        draw_indices = rng.choice(draw_indices, size=max_points, replace=False)

    uv_draw = uv_inside[draw_indices]
    depth_draw = depth_inside[draw_indices]
    order = np.argsort(depth_draw)[::-1]
    uv_draw = uv_draw[order]
    depth_draw = depth_draw[order]

    near = float(np.percentile(depth_inside, near_percentile))
    far = float(np.percentile(depth_inside, far_percentile))
    colors = depth_to_rgb(depth_draw, near, far)
    radius = max(0, int(point_radius))
    for (u_val, v_val), color in zip(uv_draw, colors):
        x_val = int(round(u_val))
        y_val = int(round(v_val))
        fill = (int(color[0]), int(color[1]), int(color[2]), 210)
        if radius == 0:
            draw.point((x_val, y_val), fill=fill)
        else:
            draw.ellipse((x_val - radius, y_val - radius, x_val + radius, y_val + radius), fill=fill)
    image.save(output_path, quality=95)
    return int(uv_draw.shape[0])


def summarize_frame(
    *,
    frame_id: str,
    image_path: Path,
    cloud_path: Path,
    points: np.ndarray,
    finite_mask: np.ndarray,
    nonzero_mask: np.ndarray,
    in_front: np.ndarray,
    inside: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    overlay_path: Path,
    drawn_points: int,
    association_row: Optional[dict],
) -> dict:
    valid_depth = depth[inside]
    uv_inside = uv[inside]
    projection_bounds = None
    if uv_inside.shape[0]:
        projection_bounds = {
            "u_min": safe_float(uv_inside[:, 0].min()),
            "u_max": safe_float(uv_inside[:, 0].max()),
            "v_min": safe_float(uv_inside[:, 1].min()),
            "v_max": safe_float(uv_inside[:, 1].max()),
        }

    finite_points = int(finite_mask.sum())
    nonzero_points = int((finite_mask & nonzero_mask).sum())
    in_front_points = int(in_front.sum())
    inside_image_points = int(inside.sum())
    dt_ns = association_row.get("image_to_cloud_dt_ns") if association_row else None
    return {
        "frame_id": frame_id,
        "image_path": str(image_path),
        "cloud_path": str(cloud_path),
        "overlay_path": str(overlay_path),
        "image_timestamp_ns": association_row.get("image_timestamp_ns") if association_row else None,
        "cloud_timestamp_ns": association_row.get("cloud_timestamp_ns") if association_row else None,
        "image_to_cloud_dt_ns": dt_ns,
        "image_to_cloud_dt_ms": safe_float(float(dt_ns) / 1e6) if dt_ns not in (None, "") else None,
        "raw_points": int(points.shape[0]),
        "finite_points": finite_points,
        "nonzero_points": nonzero_points,
        "in_front_points": in_front_points,
        "inside_image_points": inside_image_points,
        "drawn_overlay_points": int(drawn_points),
        "front_ratio": safe_float(in_front_points / max(nonzero_points, 1)),
        "coverage_ratio": safe_float(inside_image_points / max(in_front_points, 1)),
        "median_projected_depth": safe_float(np.median(valid_depth)) if valid_depth.size else None,
        "depth_min": safe_float(valid_depth.min()) if valid_depth.size else None,
        "depth_max": safe_float(valid_depth.max()) if valid_depth.size else None,
        "projection_bounds": projection_bounds,
    }


def verdict_from_summary(summary: dict, args: argparse.Namespace) -> dict:
    median_front = summary.get("median_front_ratio")
    median_coverage = summary.get("median_coverage_ratio")
    median_dt = summary.get("median_image_to_cloud_dt_ms")
    reasons = []

    if median_front is None or median_coverage is None:
        return {"level": "Error", "reasons": ["No usable projection statistics were produced."]}

    if median_front < args.warn_front_ratio:
        reasons.append(f"median_front_ratio {median_front:.3f} < {args.warn_front_ratio:.3f}")
    if median_coverage < args.warn_coverage_ratio:
        reasons.append(f"median_coverage_ratio {median_coverage:.3f} < {args.warn_coverage_ratio:.3f}")
    if median_dt is not None and median_dt > args.warn_median_dt_ms:
        reasons.append(f"median image/cloud dt {median_dt:.1f} ms > {args.warn_median_dt_ms:.1f} ms")
    if reasons:
        return {"level": "Error", "reasons": reasons}

    warning_reasons = []
    if median_front < args.good_front_ratio:
        warning_reasons.append(f"median_front_ratio {median_front:.3f} < {args.good_front_ratio:.3f}")
    if median_coverage < args.good_coverage_ratio:
        warning_reasons.append(f"median_coverage_ratio {median_coverage:.3f} < {args.good_coverage_ratio:.3f}")
    if median_dt is not None and median_dt > args.good_median_dt_ms:
        warning_reasons.append(f"median image/cloud dt {median_dt:.1f} ms > {args.good_median_dt_ms:.1f} ms")
    if warning_reasons:
        return {"level": "Warning", "reasons": warning_reasons}
    return {"level": "Good", "reasons": ["Projection coverage, front-facing ratio, and sync deltas pass v1 thresholds."]}


def run_projection_check(args: argparse.Namespace) -> dict:
    scene_dir = args.scene_dir.expanduser().resolve()
    paths = resolve_scene_paths(scene_dir)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else scene_dir / "projection_debug"
    report_path = args.report_path.expanduser().resolve() if args.report_path else paths["metadata"] / "lidar_projection_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    camera_data = load_camera_json(paths["intrinsics"])
    tf_chain = load_json(paths["tf_chain"])
    associations = load_associations(paths["associations"])
    k_mat = np.asarray(camera_data["K_like"], dtype=np.float64)
    camera_from_lidar = np.asarray(tf_chain["T_camera_from_lidar"], dtype=np.float64)
    width = int(camera_data["width"])
    height = int(camera_data["height"])

    frame_reports = []
    for frame_id in args.frames:
        image_path = find_image_path(paths["images"], frame_id)
        cloud_path = find_cloud_path(scene_dir, paths["lidar_frames"], associations, frame_id)
        points = np.load(cloud_path)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"Expected {cloud_path} to have shape Nx3 or wider, got {points.shape}")
        points = np.asarray(points[:, :3], dtype=np.float64)

        finite_mask = np.isfinite(points).all(axis=1)
        nonzero_mask = np.linalg.norm(points, axis=1) > args.min_range
        usable = finite_mask & nonzero_mask
        usable_points = points[usable]
        points_camera = transform_lidar_to_camera(usable_points, camera_from_lidar)
        uv_usable, in_front_usable = project_pinhole_k_like(points_camera, k_mat)
        depth_usable = points_camera[:, 2]
        inside_usable = (
            in_front_usable
            & (uv_usable[:, 0] >= 0)
            & (uv_usable[:, 0] < width)
            & (uv_usable[:, 1] >= 0)
            & (uv_usable[:, 1] < height)
        )

        overlay_path = output_dir / f"frame_{frame_id}_overlay.jpg"
        drawn_points = draw_overlay(
            image_path,
            overlay_path,
            uv_usable[inside_usable],
            depth_usable[inside_usable],
            args.max_overlay_points,
            args.point_radius,
            args.near_percentile,
            args.far_percentile,
        )

        full_in_front = np.zeros(points.shape[0], dtype=bool)
        full_inside = np.zeros(points.shape[0], dtype=bool)
        full_uv = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
        full_depth = np.full(points.shape[0], np.nan, dtype=np.float64)
        usable_indices = np.flatnonzero(usable)
        full_in_front[usable_indices] = in_front_usable
        full_inside[usable_indices] = inside_usable
        full_uv[usable_indices] = uv_usable
        full_depth[usable_indices] = depth_usable

        frame_report = summarize_frame(
            frame_id=frame_id,
            image_path=image_path,
            cloud_path=cloud_path,
            points=points,
            finite_mask=finite_mask,
            nonzero_mask=nonzero_mask,
            in_front=full_in_front,
            inside=full_inside,
            uv=full_uv,
            depth=full_depth,
            overlay_path=overlay_path,
            drawn_points=drawn_points,
            association_row=associations.get(frame_id),
        )
        frame_reports.append(frame_report)
        print(
            f"[INFO] {frame_id}: front={frame_report['front_ratio']:.3f} "
            f"coverage={frame_report['coverage_ratio']:.3f} overlay={overlay_path}"
        )

    coverage = np.asarray([r["coverage_ratio"] for r in frame_reports if r["coverage_ratio"] is not None], dtype=np.float64)
    front = np.asarray([r["front_ratio"] for r in frame_reports if r["front_ratio"] is not None], dtype=np.float64)
    dt_ms = np.asarray([r["image_to_cloud_dt_ms"] for r in frame_reports if r["image_to_cloud_dt_ms"] is not None], dtype=np.float64)
    summary = {
        "num_frames": len(frame_reports),
        "mean_front_ratio": safe_float(front.mean()) if front.size else None,
        "median_front_ratio": safe_float(np.median(front)) if front.size else None,
        "mean_coverage_ratio": safe_float(coverage.mean()) if coverage.size else None,
        "median_coverage_ratio": safe_float(np.median(coverage)) if coverage.size else None,
        "mean_image_to_cloud_dt_ms": safe_float(dt_ms.mean()) if dt_ms.size else None,
        "median_image_to_cloud_dt_ms": safe_float(np.median(dt_ms)) if dt_ms.size else None,
        "max_image_to_cloud_dt_ms": safe_float(dt_ms.max()) if dt_ms.size else None,
    }
    report = {
        "scene_dir": str(scene_dir),
        "projection_model": "K_like_pinhole_approximation",
        "fishpoly_support": "approximate_k_like_only",
        "note": "Current Ferrari1 FishPoly camera is checked with exported K_like pinhole approximation; full FishPoly distortion is not modeled here.",
        "camera": {
            "camera_name": camera_data.get("camera_name"),
            "camera_model": camera_data.get("camera_model"),
            "width": width,
            "height": height,
            "K_like": k_mat.tolist(),
        },
        "transform": {
            "extrinsic_direction_used": "T_camera_from_lidar",
            "T_camera_from_lidar": camera_from_lidar.tolist(),
        },
        "frames": frame_reports,
        "summary": summary,
    }
    report["verdict"] = verdict_from_summary(summary, args)
    write_json(report_path, report)
    print(f"[INFO] Projection report written: {report_path}")
    print(f"[INFO] Verdict: {report['verdict']['level']} - {'; '.join(report['verdict']['reasons'])}")
    return report


def main() -> None:
    run_projection_check(parse_args())


if __name__ == "__main__":
    main()
