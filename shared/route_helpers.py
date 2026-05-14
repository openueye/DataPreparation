from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import List

from data_preparation.shared.layout import DataPrepLayout


def invoke_module(module: str, command_name: str, argv: List[str]) -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = [command_name, *argv]
        runpy.run_module(module, run_name="__main__")
    finally:
        sys.argv = old_argv


def append_passthrough(argv: List[str], passthrough: List[str]) -> List[str]:
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return [*argv, *passthrough]


def layout_from_args(repo_root: Path | None, thesis_root: Path | None) -> DataPrepLayout:
    return DataPrepLayout.from_repo_root(repo_root=repo_root, thesis_root=thesis_root)


def resolve_route_output(
    *,
    scene: str | None,
    output_dir: Path | None,
    repo_root: Path | None,
    thesis_root: Path | None,
    category: str,
) -> Path:
    if output_dir is not None:
        return output_dir.expanduser().resolve()
    if not scene:
        raise ValueError("Pass --output-dir, or pass --scene so the default output path can be inferred.")
    layout = layout_from_args(repo_root, thesis_root)
    if category == "sfm":
        return layout.sfm_colmap_scene_dir(scene).expanduser().resolve()
    if category == "hybrid":
        return layout.hybrid_colmap_scene_dir(scene).expanduser().resolve()
    if category == "slam":
        return layout.colmap_compat_scene_dir(scene).expanduser().resolve()
    raise ValueError(f"Unsupported route category: {category}")


def scene_name_from_rosbag_dir(rosbag_dir: Path, explicit_scene: str | None = None) -> str:
    return explicit_scene or rosbag_dir.expanduser().resolve().name


def resolve_prepared_scene_output(
    *,
    scene: str,
    output_dir: Path | None,
    repo_root: Path | None,
    thesis_root: Path | None,
) -> Path:
    if output_dir is not None:
        return output_dir.expanduser().resolve()
    layout = layout_from_args(repo_root, thesis_root)
    return layout.rosbag_prepared_scene_dir(scene).expanduser().resolve()


def prepare_scene_from_rosbag(
    *,
    rosbag_dir: Path,
    scene: str,
    output_dir: Path,
    calibration: Path | None = None,
    overwrite: bool = False,
    passthrough: List[str] | None = None,
) -> Path:
    output_dir = output_dir.expanduser().resolve()
    if (output_dir / "scene_meta.json").exists() and not overwrite:
        return output_dir

    argv = [
        "--scene",
        scene,
        "--bag-dir",
        str(rosbag_dir.expanduser().resolve()),
        "--output-dir",
        str(output_dir),
    ]
    if calibration is not None:
        argv.extend(["--calibration", str(calibration.expanduser().resolve())])
    if overwrite:
        argv.append("--overwrite")
    invoke_module(
        "data_preparation.shared.organize_odin_scene",
        "organize-odin-scene",
        append_passthrough(argv, passthrough or []),
    )
    return output_dir
