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

