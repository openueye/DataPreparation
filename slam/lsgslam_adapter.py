from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

try:
    from data_preparation.shared.io import load_json, require_cv2, write_json
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from data_preparation.shared.io import load_json, require_cv2, write_json


def qvec_to_rotmat(qvec: Iterable[float]) -> np.ndarray:
    qw, qx, qy, qz = [float(value) for value in qvec]
    return np.asarray(
        [
            [
                1.0 - 2.0 * qy * qy - 2.0 * qz * qz,
                2.0 * qx * qy - 2.0 * qz * qw,
                2.0 * qz * qx + 2.0 * qy * qw,
            ],
            [
                2.0 * qx * qy + 2.0 * qz * qw,
                1.0 - 2.0 * qx * qx - 2.0 * qz * qz,
                2.0 * qy * qz - 2.0 * qx * qw,
            ],
            [
                2.0 * qz * qx - 2.0 * qy * qw,
                2.0 * qy * qz + 2.0 * qx * qw,
                1.0 - 2.0 * qx * qx - 2.0 * qy * qy,
            ],
        ],
        dtype=np.float64,
    )


def read_colmap_image_entries(images_txt: Path) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    for raw_line in images_txt.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        image_id = int(parts[0])
        qvec = np.asarray([float(value) for value in parts[1:5]], dtype=np.float64)
        tvec = np.asarray([float(value) for value in parts[5:8]], dtype=np.float64)
        image_name = parts[9]
        stem = Path(image_name).stem
        rotation_camera_from_world = qvec_to_rotmat(qvec)
        world_from_camera = np.eye(4, dtype=np.float64)
        world_from_camera[:3, :3] = rotation_camera_from_world.T
        world_from_camera[:3, 3] = -rotation_camera_from_world.T @ tvec
        entries.append(
            {
                "image_id": image_id,
                "image_name": image_name,
                "stem": stem,
                "world_from_camera": world_from_camera,
            }
        )
    return sorted(entries, key=lambda item: int(item["image_id"]))


def _format_float(value: float) -> str:
    return f"{float(value):.12g}"


def write_traj_txt(path: Path, entries: Iterable[Dict[str, object]]) -> None:
    lines = []
    for entry in entries:
        c2w = np.asarray(entry["world_from_camera"], dtype=np.float64)[:3, :4]
        values = " ".join(_format_float(value) for value in c2w.reshape(-1))
        lines.append(f"{entry['stem']} {values}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_image_to_png(source_path: Path, target_path: Path) -> None:
    cv2 = require_cv2("LSG-SLAM image conversion")
    image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read source image: {source_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target_path), image):
        raise ValueError(f"Failed to write PNG image: {target_path}")


def link_or_copy_depth(source_path: Path, target_path: Path, *, copy_depths: bool) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()
    if copy_depths:
        shutil.copy2(source_path, target_path)
    else:
        target_path.symlink_to(source_path.resolve())


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _maybe_load_json(path: Path) -> Optional[object]:
    return load_json(path) if path.exists() else None


def format_segment_run_name(scene_name: str, start: int, end: int, stride: int) -> str:
    if start < 0:
        raise ValueError(f"start must be non-negative, got {start}")
    if end <= start:
        raise ValueError(f"end must be greater than start, got start={start}, end={end}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    return f"{scene_name}_{start:06d}_{end:06d}_{stride}"


def plan_lsgslam_segments(total_frames: int, segment_length: int, stride: int = 1) -> List[Dict[str, int]]:
    """Plan inclusive LSG-SLAM odometry segments with shared boundary frames."""
    if total_frames <= 1:
        raise ValueError(f"total_frames must be greater than 1, got {total_frames}")
    if segment_length <= 1:
        raise ValueError(f"segment_length must be greater than 1, got {segment_length}")
    if segment_length >= total_frames:
        raise ValueError(
            f"segment_length must be smaller than total_frames for segmented LSG-SLAM runs, "
            f"got segment_length={segment_length}, total_frames={total_frames}"
        )
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    last_index = total_frames - 1
    segments: List[Dict[str, int]] = []
    start = 0
    while start < last_index:
        end = min(start + segment_length, last_index)
        segments.append({"start": start, "end": end, "stride": stride})
        if end == last_index:
            break
        start = end
    return segments


def render_lsgslam_segment_config(
    *,
    base_config_import: str,
    scene_name: str,
    output_root: str,
    start: int,
    end: int,
    stride: int,
    progress_every: int = 100,
) -> str:
    run_name = format_segment_run_name(scene_name, start, end, stride)
    if progress_every <= 0:
        raise ValueError(f"progress_every must be positive, got {progress_every}")
    return f'''from __future__ import annotations

from copy import deepcopy

from {base_config_import} import config as base_config


start_idx = {start}
end_idx = {end}
stride = {stride}

scene_name = "{scene_name}"
group_name = "{output_root}"
run_name = f"{{scene_name}}_{{start_idx:06d}}_{{end_idx:06d}}_{{stride}}"

config = deepcopy(base_config)
config["workdir"] = group_name
config["run_name"] = run_name
config["scene_path"] = ""
config["report_global_progress_every"] = {progress_every}
config["run_loop_closure"] = True
config["data"]["name"] = run_name
config["data"]["sequence"] = scene_name
config["data"]["start"] = start_idx
config["data"]["end"] = end_idx
config["data"]["stride"] = stride
config["data"]["num_frames"] = -1
config["wandb"]["group"] = "downtown1_lsgslam_paper_pipeline"
config["wandb"]["name"] = run_name
'''


def parse_segment_run_name(run_name: str) -> Dict[str, object]:
    parts = run_name.split("_")
    if len(parts) < 4:
        raise ValueError(f"Run name must end with numeric start/end/stride suffix: {run_name}")
    start_raw, end_raw, stride_raw = parts[-3:]
    if not (start_raw.isdigit() and end_raw.isdigit() and stride_raw.isdigit()):
        raise ValueError(f"Run name must end with numeric start/end/stride suffix: {run_name}")
    scene_prefix = "_".join(parts[:-3])
    if not scene_prefix:
        raise ValueError(f"Run name must include a scene prefix before numeric start/end/stride suffix: {run_name}")
    start = int(start_raw)
    end = int(end_raw)
    stride = int(stride_raw)
    if end <= start:
        raise ValueError(f"Segment end must be greater than start in run name: {run_name}")
    if stride <= 0:
        raise ValueError(f"Segment stride must be positive in run name: {run_name}")
    return {"scene_prefix": scene_prefix, "start": start, "end": end, "stride": stride}


def _sorted_stems(directory: Path, suffix: str) -> List[str]:
    return sorted(path.stem for path in directory.glob(f"*{suffix}") if path.is_file() or path.is_symlink())


def validate_lsgslam_feature_layout(scene_dir: Path) -> Dict[str, object]:
    scene_dir = scene_dir.expanduser().resolve()
    data_rect_dir = scene_dir / "data_rect"
    depth_dir = scene_dir / "depth_sceneflow"
    feature_dir = scene_dir / "global_features"
    traj_path = scene_dir / "traj.txt"
    for required in (data_rect_dir, depth_dir, feature_dir, traj_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing LSG-SLAM layout path: {required}")

    rgb_stems = _sorted_stems(data_rect_dir, ".png")
    depth_stems = _sorted_stems(depth_dir, ".npy")
    feature_stems = _sorted_stems(feature_dir, ".npy")
    traj_stems = [
        line.split()[0]
        for line in traj_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rgb_stems:
        raise ValueError(f"No RGB PNG frames found under {data_rect_dir}")
    expected = rgb_stems
    mismatches = {
        "depth": depth_stems,
        "global_features": feature_stems,
        "traj": traj_stems,
    }
    for label, stems in mismatches.items():
        if stems != expected:
            raise ValueError(
                f"LSG-SLAM {label} stems do not match RGB stems in {scene_dir}: "
                f"rgb_count={len(expected)}, {label}_count={len(stems)}"
            )

    return {
        "scene_dir": str(scene_dir),
        "rgb_count": len(rgb_stems),
        "depth_count": len(depth_stems),
        "feature_count": len(feature_stems),
        "traj_count": len(traj_stems),
        "first_stem": rgb_stems[0],
        "last_stem": rgb_stems[-1],
    }


def export_lsgslam_euroc_scene(
    source_scene: Path,
    output_dir: Path,
    *,
    depths_dir: Path,
    copy_images: bool = True,
    copy_depths: bool = False,
    overwrite: bool = False,
) -> Dict[str, object]:
    source_scene = source_scene.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    depths_dir = depths_dir.expanduser().resolve()

    sparse_dir = source_scene / "sparse" / "0"
    images_dir = source_scene / "images"
    images_txt = sparse_dir / "images.txt"
    cameras_txt = sparse_dir / "cameras.txt"
    for required in (images_dir, images_txt, cameras_txt, depths_dir):
        if not required.exists():
            raise FileNotFoundError(f"Missing required input: {required}")

    entries = read_colmap_image_entries(images_txt)
    if not entries:
        raise ValueError(f"No COLMAP image entries found: {images_txt}")

    _prepare_output_dir(output_dir, overwrite=overwrite)
    data_rect_dir = output_dir / "data_rect"
    depth_output_dir = output_dir / "depth_sceneflow"

    missing: List[str] = []
    for entry in entries:
        stem = str(entry["stem"])
        source_image = images_dir / str(entry["image_name"])
        source_depth = depths_dir / f"{stem}.npy"
        if not source_image.exists():
            missing.append(str(source_image))
        if not source_depth.exists():
            missing.append(str(source_depth))
    if missing:
        raise FileNotFoundError("Missing LSG-SLAM adapter inputs:\n" + "\n".join(missing[:20]))

    for entry in entries:
        stem = str(entry["stem"])
        source_image = images_dir / str(entry["image_name"])
        source_depth = depths_dir / f"{stem}.npy"
        convert_image_to_png(source_image, data_rect_dir / f"{stem}.png")
        link_or_copy_depth(source_depth, depth_output_dir / f"{stem}.npy", copy_depths=copy_depths)

    write_traj_txt(output_dir / "traj.txt", entries)
    metadata: Dict[str, object] = {
        "format": "lsgslam_euroc_style",
        "source_scene": str(source_scene),
        "output_dir": str(output_dir),
        "images_dir": str(images_dir),
        "depths_dir": str(depths_dir),
        "frame_count": len(entries),
        "image_output": {"directory": "data_rect", "extension": ".png", "mode": "convert_copy" if copy_images else "convert"},
        "depth_output": {"directory": "depth_sceneflow", "extension": ".npy", "mode": "copy" if copy_depths else "symlink"},
        "traj_format": "stem followed by row-major 3x4 camera-to-world matrix from COLMAP inverse pose",
        "camera_rectified": _maybe_load_json(source_scene / "metadata" / "camera_rectified.json"),
        "cameras_txt": str(cameras_txt),
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def run(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export a COLMAP-compatible SLAM scene to LSG-SLAM EuRoC-style RGB-D layout.")
    parser.add_argument("--source-scene", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--depths-dir", required=True, type=Path)
    parser.add_argument("--copy-depths", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    metadata = export_lsgslam_euroc_scene(
        args.source_scene,
        args.output_dir,
        depths_dir=args.depths_dir,
        copy_depths=args.copy_depths,
        overwrite=args.overwrite,
    )
    print(f"Wrote {metadata['frame_count']} LSG-SLAM frames to {metadata['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
