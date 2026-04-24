#!/usr/bin/env python3
"""Convert a video or prepared image directory into a COLMAP dataset usable by this repo."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

cv2 = None


def require_cv2():
    global cv2
    if cv2 is None:
        try:
            import cv2 as cv2_module
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "OpenCV is required for video preprocessing. Install opencv-python or opencv-contrib-python."
            ) from exc
        cv2 = cv2_module
    return cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frames from a video or use a prepared image directory, run COLMAP, undistort the reconstruction, "
            "and produce a dataset layout compatible with this 3DGS repository."
        )
    )
    parser.add_argument("--video-path", help="Input video path.")
    parser.add_argument(
        "--image-dir",
        help="Prepared input image directory. When set, video extraction is skipped.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Final dataset folder. It will contain images/ and sparse/0/.",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=3.0,
        help="Target frame sampling fps. Set <= 0 to disable fps-based sampling.",
    )
    parser.add_argument(
        "--every-nth-frame",
        type=int,
        default=0,
        help="Alternative to sample-fps. Keep every Nth frame. Set 0 to ignore.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional hard cap on extracted frames. Set 0 to keep all sampled frames.",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=1600,
        help="Resize longer side to this value before saving. Set 0 to keep original size.",
    )
    parser.add_argument(
        "--image-format",
        choices=("jpg", "png"),
        default="jpg",
        help="Saved frame format.",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality when image-format=jpg.",
    )
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=0.0,
        help="Drop blurry frames using Laplacian variance threshold. Set 0 to disable.",
    )
    parser.add_argument(
        "--matcher",
        choices=("sequential", "exhaustive"),
        default="sequential",
        help="COLMAP matcher. sequential is recommended for videos.",
    )
    parser.add_argument(
        "--sequential-overlap",
        type=int,
        default=10,
        help="Neighbor overlap for COLMAP sequential matcher.",
    )
    parser.add_argument(
        "--camera-model",
        choices=("SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "OPENCV"),
        default="OPENCV",
        help=(
            "Camera model used during feature extraction. The final exported dataset "
            "will be undistorted by COLMAP into PINHOLE/SIMPLE_PINHOLE."
        ),
    )
    parser.add_argument(
        "--camera-params",
        default="",
        help=(
            "Optional COLMAP camera params string, e.g. fx,fy,cx,cy or fx,cx,cy. "
            "Leave empty to let COLMAP initialize automatically."
        ),
    )
    parser.add_argument(
        "--default-focal-length-factor",
        type=float,
        default=1.2,
        help="Fallback focal length factor for images without EXIF.",
    )
    parser.add_argument(
        "--colmap-path",
        default="colmap",
        help="COLMAP executable path.",
    )
    parser.add_argument(
        "--sift-gpu",
        action="store_true",
        help="Enable GPU for SIFT extraction/matching if COLMAP supports it.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output-dir if it already exists.",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep temporary raw COLMAP workspace under .cache_preprocess/.",
    )
    return parser.parse_args()


def ensure_binary(name_or_path: str) -> str:
    resolved = shutil.which(name_or_path) if os.path.sep not in name_or_path else name_or_path
    if resolved is None or not Path(resolved).exists():
        raise FileNotFoundError(f"Cannot find executable: {name_or_path}")
    return resolved


def run_command(cmd: List[str]) -> None:
    print("\n[RUN]", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


_COLMAP_HELP_CACHE: Dict[str, str] = {}


def get_colmap_help(colmap_path: str, command: str) -> str:
    cache_key = f"{colmap_path}::{command}"
    if cache_key in _COLMAP_HELP_CACHE:
        return _COLMAP_HELP_CACHE[cache_key]

    result = subprocess.run(
        [colmap_path, command, "-h"],
        check=False,
        capture_output=True,
        text=True,
    )
    help_text = (result.stdout or "") + "\n" + (result.stderr or "")
    if result.returncode != 0 or not help_text.strip():
        raise RuntimeError(f"Failed to query COLMAP help for command: {command}")

    _COLMAP_HELP_CACHE[cache_key] = help_text
    return help_text


def resolve_colmap_option(colmap_path: str, command: str, candidates: List[str]) -> str:
    help_text = get_colmap_help(colmap_path, command)
    for option_name in candidates:
        if option_name in help_text:
            return option_name
    raise RuntimeError(
        f"None of the candidate options are supported by '{command}': {', '.join(candidates)}"
    )


def resize_if_needed(image, max_image_size: int):
    if max_image_size <= 0:
        return image
    height, width = image.shape[:2]
    long_side = max(height, width)
    if long_side <= max_image_size:
        return image
    scale = max_image_size / float(long_side)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def laplacian_sharpness(image) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def build_sampling_step(video_fps: float, sample_fps: float, every_nth_frame: int) -> int:
    if every_nth_frame and every_nth_frame > 0:
        return every_nth_frame
    if sample_fps and sample_fps > 0 and video_fps and video_fps > 0:
        return max(int(round(video_fps / sample_fps)), 1)
    return 1


def extract_frames(
    video_path: Path,
    image_dir: Path,
    image_format: str,
    jpg_quality: int,
    sample_fps: float,
    every_nth_frame: int,
    max_frames: int,
    max_image_size: int,
    min_sharpness: float,
) -> int:
    require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    step = build_sampling_step(video_fps, sample_fps, every_nth_frame)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] video fps={video_fps:.3f}, total_frames={total_frames}, sampling_step={step}")

    suffix = ".jpg" if image_format == "jpg" else ".png"
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality] if image_format == "jpg" else []

    saved = 0
    seen = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if seen % step != 0:
            seen += 1
            continue

        frame = resize_if_needed(frame, max_image_size)
        if min_sharpness > 0.0 and laplacian_sharpness(frame) < min_sharpness:
            seen += 1
            continue

        filename = f"frame_{saved + 1:05d}{suffix}"
        out_path = image_dir / filename
        if not cv2.imwrite(str(out_path), frame, encode_params):
            raise RuntimeError(f"Failed to write frame: {out_path}")

        saved += 1
        seen += 1
        if max_frames > 0 and saved >= max_frames:
            break

    cap.release()
    if saved < 8:
        raise RuntimeError(
            f"Only {saved} frames were extracted. COLMAP usually needs more overlap and more views."
        )
    print(f"[INFO] extracted {saved} frames into {image_dir}")
    return saved


def find_sparse_model_dir(sparse_root: Path) -> Path:
    candidates = [path for path in sparse_root.iterdir() if path.is_dir()]
    if not candidates:
        raise RuntimeError(f"No sparse models found under: {sparse_root}")

    def score(path: Path) -> int:
        images_bin = path / "images.bin"
        return images_bin.stat().st_size if images_bin.exists() else -1

    best = max(candidates, key=score)
    if score(best) <= 0:
        raise RuntimeError(f"Could not find a valid sparse model in: {sparse_root}")
    return best


def ensure_sparse_zero_layout(output_dir: Path) -> None:
    sparse_dir = output_dir / "sparse"
    if not sparse_dir.exists():
        raise RuntimeError(f"COLMAP did not produce sparse output under: {sparse_dir}")

    zero_dir = sparse_dir / "0"
    if zero_dir.exists():
        return

    bin_files = ["cameras.bin", "images.bin", "points3D.bin"]
    txt_files = ["cameras.txt", "images.txt", "points3D.txt"]
    has_bin = all((sparse_dir / name).exists() for name in bin_files)
    has_txt = all((sparse_dir / name).exists() for name in txt_files)
    if not has_bin and not has_txt:
        raise RuntimeError(f"Unexpected sparse folder layout in: {sparse_dir}")

    zero_dir.mkdir(parents=True, exist_ok=True)
    for name in bin_files + txt_files:
        src = sparse_dir / name
        if src.exists():
            shutil.move(str(src), str(zero_dir / name))


def colmap_feature_extractor(
    colmap_path: str,
    database_path: Path,
    image_path: Path,
    camera_model: str,
    camera_params: str,
    default_focal_length_factor: float,
    use_gpu: bool,
) -> None:
    use_gpu_option = resolve_colmap_option(
        colmap_path,
        "feature_extractor",
        ["--FeatureExtraction.use_gpu", "--SiftExtraction.use_gpu"],
    )
    cmd = [
        colmap_path,
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--ImageReader.single_camera",
        "1",
        "--ImageReader.camera_model",
        camera_model,
        "--ImageReader.default_focal_length_factor",
        str(default_focal_length_factor),
        use_gpu_option,
        "1" if use_gpu else "0",
    ]
    if camera_params:
        cmd.extend(["--ImageReader.camera_params", camera_params])
    run_command(cmd)


def colmap_matcher(
    colmap_path: str,
    matcher: str,
    database_path: Path,
    sequential_overlap: int,
    use_gpu: bool,
) -> None:
    use_gpu_option = resolve_colmap_option(
        colmap_path,
        f"{matcher}_matcher",
        ["--FeatureMatching.use_gpu", "--SiftMatching.use_gpu"],
    )
    if matcher == "sequential":
        cmd = [
            colmap_path,
            "sequential_matcher",
            "--database_path",
            str(database_path),
            "--SequentialMatching.overlap",
            str(sequential_overlap),
            use_gpu_option,
            "1" if use_gpu else "0",
        ]
    else:
        cmd = [
            colmap_path,
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
            use_gpu_option,
            "1" if use_gpu else "0",
        ]
    run_command(cmd)


def colmap_mapper(colmap_path: str, database_path: Path, image_path: Path, sparse_path: Path) -> None:
    cmd = [
        colmap_path,
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--output_path",
        str(sparse_path),
    ]
    run_command(cmd)


def colmap_undistort(
    colmap_path: str,
    image_path: Path,
    sparse_model_path: Path,
    output_dir: Path,
) -> None:
    cmd = [
        colmap_path,
        "image_undistorter",
        "--image_path",
        str(image_path),
        "--input_path",
        str(sparse_model_path),
        "--output_path",
        str(output_dir),
        "--output_type",
        "COLMAP",
    ]
    run_command(cmd)


def remove_if_exists(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
            )
        remove_if_exists([output_dir])
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()

    if bool(args.video_path) == bool(args.image_dir):
        raise ValueError("Pass exactly one of --video-path or --image-dir.")

    video_path = Path(args.video_path).expanduser().resolve() if args.video_path else None
    input_image_dir = Path(args.image_dir).expanduser().resolve() if args.image_dir else None
    output_dir = Path(args.output_dir).expanduser().resolve()
    if video_path is not None and not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    if input_image_dir is not None and not input_image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {input_image_dir}")
    if input_image_dir is not None and not input_image_dir.is_dir():
        raise NotADirectoryError(f"Image input is not a directory: {input_image_dir}")

    colmap_path = ensure_binary(args.colmap_path)
    prepare_output_dir(output_dir, overwrite=args.overwrite)

    cache_dir = output_dir / ".cache_preprocess"
    raw_image_dir = input_image_dir if input_image_dir is not None else cache_dir / "raw_images"
    database_path = cache_dir / "database.db"
    raw_sparse_dir = cache_dir / "raw_sparse"

    if input_image_dir is None:
        raw_image_dir.mkdir(parents=True, exist_ok=True)
    raw_sparse_dir.mkdir(parents=True, exist_ok=True)

    if video_path is not None:
        extract_frames(
            video_path=video_path,
            image_dir=raw_image_dir,
            image_format=args.image_format,
            jpg_quality=args.jpg_quality,
            sample_fps=args.sample_fps,
            every_nth_frame=args.every_nth_frame,
            max_frames=args.max_frames,
            max_image_size=args.max_image_size,
            min_sharpness=args.min_sharpness,
        )
    else:
        image_count = len([path for path in raw_image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if image_count < 8:
            raise RuntimeError(f"Only {image_count} images found in {raw_image_dir}. COLMAP usually needs more views.")
        print(f"[INFO] using {image_count} prepared images from {raw_image_dir}")

    colmap_feature_extractor(
        colmap_path=colmap_path,
        database_path=database_path,
        image_path=raw_image_dir,
        camera_model=args.camera_model,
        camera_params=args.camera_params,
        default_focal_length_factor=args.default_focal_length_factor,
        use_gpu=args.sift_gpu,
    )
    colmap_matcher(
        colmap_path=colmap_path,
        matcher=args.matcher,
        database_path=database_path,
        sequential_overlap=args.sequential_overlap,
        use_gpu=args.sift_gpu,
    )
    colmap_mapper(
        colmap_path=colmap_path,
        database_path=database_path,
        image_path=raw_image_dir,
        sparse_path=raw_sparse_dir,
    )

    selected_model = find_sparse_model_dir(raw_sparse_dir)
    print(f"[INFO] selected sparse model: {selected_model}")
    colmap_undistort(
        colmap_path=colmap_path,
        image_path=raw_image_dir,
        sparse_model_path=selected_model,
        output_dir=output_dir,
    )

    ensure_sparse_zero_layout(output_dir)

    if not args.keep_intermediate:
        remove_if_exists([cache_dir, output_dir / "stereo", output_dir / "run-colmap-geometric.sh", output_dir / "run-colmap-photometric.sh"])

    print("\n[OK] Dataset is ready.")
    print(f"[OK] Use this folder as the training --data-dir: {output_dir}")
    print(f"[OK] Images: {output_dir / 'images'}")
    print(f"[OK] Sparse model: {output_dir / 'sparse' / '0'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
