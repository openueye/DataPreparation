from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from data_preparation.shared.io import load_json, write_json
from data_preparation.shared.layout import DataPrepLayout
from data_preparation.shared.presets import PRESETS, get_preset
from data_preparation.shared.reports import write_workflow_report


def _invoke_module(module: str, argv: List[str]) -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = [module.rsplit(".", 1)[-1], *argv]
        runpy.run_module(module, run_name="__main__")
    finally:
        sys.argv = old_argv


def _layout_from_args(args: argparse.Namespace) -> DataPrepLayout:
    return DataPrepLayout.from_repo_root(repo_root=args.repo_root, thesis_root=args.thesis_root)


def _append_passthrough(argv: List[str], passthrough: List[str]) -> List[str]:
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return [*argv, *passthrough]


def _path_has_contents(path: Path) -> bool:
    if not path.exists():
        return False
    if not path.is_dir():
        return True
    return any(path.iterdir())


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def _require_empty_or_overwrite(path: Path, overwrite: bool) -> None:
    if _path_has_contents(path) and not overwrite:
        raise FileExistsError(f"Output directory already exists and is non-empty: {path}. Use --overwrite to proceed.")


def _pinhole_camera_params(scene_dir: Path) -> str:
    camera_data = load_json(scene_dir / "intrinsics" / "camera.json")
    k_like = camera_data["K_like"]
    fx = float(k_like[0][0])
    fy = float(k_like[1][1])
    cx = float(k_like[0][2])
    cy = float(k_like[1][2])
    return f"{fx},{fy},{cx},{cy}"


def _print_dry_run_summary(
    *,
    command: str,
    scene: str,
    preset_name: str,
    inputs: Dict[str, Path],
    outputs: Dict[str, Path],
    preset_args: Dict[str, Any],
    backend_argv: List[str],
    passthrough: List[str],
) -> None:
    print(f"[DRY-RUN] data_preparation {command}")
    print(f"[DRY-RUN] scene={scene} preset={preset_name}")
    print("[DRY-RUN] Inputs:")
    for key, value in inputs.items():
        print(f"  - {key}: {value} exists={value.exists()}")
    print("[DRY-RUN] Outputs:")
    for key, value in outputs.items():
        state = "missing"
        if value.exists():
            state = "non-empty" if value.is_dir() and _path_has_contents(value) else "exists"
        print(f"  - {key}: {value} state={state}")
    print("[DRY-RUN] Preset-injected parameters:")
    for key, value in preset_args.items():
        print(f"  - {key}: {value}")
    print("[DRY-RUN] Backend argv:")
    print("  " + " ".join(backend_argv))
    if passthrough:
        print("[DRY-RUN] Passthrough argv:")
        print("  " + " ".join(passthrough[1:] if passthrough[0] == "--" else passthrough))


def _write_scene_inspect_summary(summary_json: Path, scene: str, output_path: Path) -> Optional[Path]:
    if not summary_json.exists():
        return None
    summary = load_json(summary_json)
    scene_bags = [bag for bag in summary.get("bags", []) if bag.get("name") == scene]
    payload = {
        "scene": scene,
        "source_summary": str(summary_json),
        "matched_bag_count": len(scene_bags),
        "bags": scene_bags,
    }
    write_json(output_path, payload)
    return output_path


def _bag_summary_payload(bag: Any) -> Dict[str, Any]:
    return _jsonable({
        "name": bag.name,
        "bag_dir": str(bag.bag_dir),
        "db_path": str(bag.db_path) if bag.db_path else None,
        "metadata_path": str(bag.metadata_path) if bag.metadata_path else None,
        "size_bytes": bag.size_bytes,
        "ros_distro": bag.ros_distro,
        "valid": bag.valid,
        "error": bag.error,
        "start_ns": bag.start_ns,
        "end_ns": bag.end_ns,
        "duration_ns": bag.duration_ns,
        "total_messages": bag.total_messages,
        "topics": bag.topics,
        "sync_stats": bag.sync_stats,
    })


def _write_scene_feasibility_report(path: Path, scene: str, bag: Any, calibration_path: Path) -> None:
    lines = [
        f"# Scene Feasibility Report: {scene}",
        "",
        f"- Requested scene: `{scene}`",
        f"- Scene bag directory: `{bag.bag_dir}`",
        f"- Calibration file: `{calibration_path}`",
        f"- Bag status: `{'OK' if bag.valid else 'Broken'}`",
    ]
    if bag.error:
        lines.append(f"- Error: `{bag.error}`")
    if bag.valid:
        topic_names = {topic["name"] for topic in bag.topics}
        required_topics = {
            "image": "/odin1/image/compressed",
            "odometry": "/odin1/odometry",
            "raw_lidar": "/odin1/cloud_raw",
        }
        lines.extend(
            [
                f"- Total messages: `{bag.total_messages}`",
                f"- Duration ns: `{bag.duration_ns}`",
                "",
                "## Required Topics",
                "",
            ]
        )
        for label, topic in required_topics.items():
            lines.append(f"- {label}: `{'present' if topic in topic_names else 'missing'}` ({topic})")
        if bag.sync_stats:
            lines.extend(["", "## Time Alignment", ""])
            for key, stats in bag.sync_stats.items():
                if stats:
                    lines.append(
                        f"- `{key}`: matched={stats['matched_count']}, "
                        f"mean={stats['mean_ms']:.3f} ms, median={stats['median_ms']:.3f} ms, "
                        f"p95={stats['p95_ms']:.3f} ms"
                    )
    lines.extend(
        [
            "",
            "## Scope Note",
            "",
            "This workflow report is scene-scoped. The legacy `rosbag-inspect` command remains available for full-root scans.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser(command: str) -> argparse.ArgumentParser:
    preset_choices = sorted(PRESETS)
    parser = argparse.ArgumentParser(prog=f"python -m data_preparation {command}")
    scene_help = "Logical scene name, e.g. Ferrari1."
    if command == "inspect":
        scene_help = "Logical scene name. The workflow scans that scene's bag directory and writes scene-scoped reports."
    parser.add_argument("--scene", required=True, help=scene_help)
    parser.add_argument("--preset", choices=preset_choices, default="full", help="Workflow preset.")
    parser.add_argument("--repo-root", type=Path, default=None, help="Optional 00_Baselines repo root.")
    parser.add_argument("--thesis-root", type=Path, default=None, help="Optional explicit Thesis root.")
    if command in {"prepare", "run"}:
        parser.add_argument(
            "--source",
            choices=("rosbag", "rosbag-sfm", "video"),
            default="rosbag",
            help="Raw input source type.",
        )
        parser.add_argument("--video-path", type=Path, default=None, help="Required for video unless it can be inferred.")
        parser.add_argument("--overwrite", action="store_true", help="Allow prepare to reuse an existing non-empty output directory.")
    if command == "prepare":
        parser.add_argument("--dry-run", action="store_true", help="Print inferred inputs, outputs, and backend args without running backend.")
    if command == "export":
        parser.add_argument(
            "--format",
            choices=("colmap-compatible",),
            default="colmap-compatible",
            help="Export format.",
        )
        parser.add_argument(
            "--rectified",
            action="store_true",
            help="Export the scene's FishPoly-to-pinhole rectified variant.",
        )
    if command == "run":
        parser.add_argument("--with-inspect", action="store_true", help="Run inspect before prepare.")
    if command not in {"inspect", "run"}:
        parser.add_argument("passthrough", nargs=argparse.REMAINDER, help="Advanced args passed to the backend after --.")
    return parser


def cmd_inspect(args: argparse.Namespace) -> int:
    layout = _layout_from_args(args)
    output_dir = layout.validation_task_dir(args.scene, "inspect")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "inspection_summary.json"
    scene_summary = output_dir / "scene_summary.json"
    feasibility = output_dir / "feasibility_report.md"

    from data_preparation.rosbag_to_3dgs.inspect_rosbags_for_3dgs import (
        build_bag_inventory_markdown,
        parse_calibration,
        scan_bag_dir,
        summarize_dataset,
    )

    calibration = parse_calibration(layout.calibration_file())
    bag = scan_bag_dir(layout.bag_dir(args.scene))
    bags = [bag]
    (output_dir / "bag_inventory.md").write_text(
        build_bag_inventory_markdown(bags, calibration, layout.bag_dir(args.scene)),
        encoding="utf-8",
    )
    _write_scene_feasibility_report(feasibility, args.scene, bag, layout.calibration_file())
    write_json(
        summary_json,
        {
            "calibration": _jsonable(calibration),
            "dataset_summary": _jsonable(summarize_dataset(bags, calibration)),
            "bags": [_bag_summary_payload(bag)],
        },
    )
    _write_scene_inspect_summary(summary_json, args.scene, scene_summary)
    report = write_workflow_report(
        layout=layout,
        scene=args.scene,
        task="inspect",
        command="inspect",
        preset=args.preset,
        outputs={
            "inventory": output_dir / "bag_inventory.md",
            "feasibility": output_dir / "feasibility_report.md",
            "summary": summary_json,
            "scene_summary": scene_summary,
        },
        backend_args={"scan_bag_dir": layout.bag_dir(args.scene)},
        scope={
            "user_requested_scene": args.scene,
            "workflow_scan_scope": layout.bag_dir(args.scene),
            "legacy_backend_default_scan_root": layout.rosbag_root,
        },
    )
    print(f"[INFO] inspect report={report}")
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    layout = _layout_from_args(args)
    preset = get_preset(args.preset)
    dry_run = getattr(args, "dry_run", False)
    if args.source == "rosbag":
        output_dir = layout.lidar_scene_dir_for_preset(args.scene, args.preset)
        backend_argv = [
            "--bag-dir",
            str(layout.bag_dir(args.scene)),
            "--output-dir",
            str(output_dir),
            "--calibration",
            str(layout.calibration_file()),
            "--export-cloud-frames",
        ]
        if preset.limit_images is not None:
            backend_argv.extend(["--limit-images", str(preset.limit_images)])
        if dry_run:
            _print_dry_run_summary(
                command="prepare",
                scene=args.scene,
                preset_name=args.preset,
                inputs={"bag_dir": layout.bag_dir(args.scene), "calibration": layout.calibration_file()},
                outputs={"scene_dir": output_dir},
                preset_args={
                    "source": "rosbag",
                    "limit_images": preset.limit_images,
                    "output_scene_name": output_dir.name,
                    "export_cloud_frames": True,
                },
                backend_argv=backend_argv,
                passthrough=args.passthrough,
            )
            return 0
        _require_empty_or_overwrite(output_dir, args.overwrite)
        _invoke_module(
            "data_preparation.rosbag_to_3dgs.convert_rosbag_to_3dgs",
            _append_passthrough(backend_argv, args.passthrough),
        )
        outputs = {"scene_dir": output_dir}
        backend_args = {
            "source": "rosbag",
            "bag_dir": layout.bag_dir(args.scene),
            "calibration": layout.calibration_file(),
            "preset": args.preset,
            "overwrite": args.overwrite,
        }
    elif args.source == "video":
        video_path = layout.resolve_video_path(args.scene, args.video_path)
        output_dir = layout.colmap_scene_dir(args.scene)
        backend_argv = [
            "--video-path",
            str(video_path),
            "--output-dir",
            str(output_dir),
        ]
        if preset.video_max_frames > 0:
            backend_argv.extend(["--max-frames", str(preset.video_max_frames)])
        if args.overwrite:
            backend_argv.append("--overwrite")
        if dry_run:
            _print_dry_run_summary(
                command="prepare",
                scene=args.scene,
                preset_name=args.preset,
                inputs={"video_path": video_path},
                outputs={"scene_dir": output_dir},
                preset_args={"source": "video", "video_max_frames": preset.video_max_frames},
                backend_argv=backend_argv,
                passthrough=args.passthrough,
            )
            return 0
        _require_empty_or_overwrite(output_dir, args.overwrite)
        _invoke_module(
            "data_preparation.video2colmap.preprocess_video_to_colmap",
            _append_passthrough(backend_argv, args.passthrough),
        )
        outputs = {"scene_dir": output_dir}
        backend_args = {"source": "video", "video_path": video_path, "preset": args.preset, "overwrite": args.overwrite}
    else:
        output_dir = layout.sfm_colmap_scene_dir(args.scene)
        staging_dir = layout.validation_task_dir(args.scene, "rosbag_sfm")
        raw_scene_dir = staging_dir / "raw_fishpoly_scene"
        rectified_scene_dir = staging_dir / "pinhole_rectified_scene"
        extract_argv = [
            "--bag-dir",
            str(layout.bag_dir(args.scene)),
            "--output-dir",
            str(raw_scene_dir),
            "--calibration",
            str(layout.calibration_file()),
            "--overwrite",
        ]
        if preset.limit_images is not None:
            extract_argv.extend(["--limit-images", str(preset.limit_images)])
        rectify_argv = [
            "--scene-dir",
            str(raw_scene_dir),
            "--output-dir",
            str(rectified_scene_dir),
            "--overwrite",
        ]
        colmap_argv = [
            "--image-dir",
            str(rectified_scene_dir / "images"),
            "--output-dir",
            str(output_dir),
            "--camera-model",
            "PINHOLE",
            "--matcher",
            "sequential",
        ]
        if args.overwrite:
            colmap_argv.append("--overwrite")
        if dry_run:
            _print_dry_run_summary(
                command="prepare",
                scene=args.scene,
                preset_name=args.preset,
                inputs={"bag_dir": layout.bag_dir(args.scene), "calibration": layout.calibration_file()},
                outputs={
                    "staging_raw_scene": raw_scene_dir,
                    "staging_rectified_scene": rectified_scene_dir,
                    "scene_dir": output_dir,
                },
                preset_args={
                    "source": "rosbag-sfm",
                    "limit_images": preset.limit_images,
                    "output_scene_name": output_dir.name,
                    "uses_odometry": False,
                    "uses_lidar": False,
                },
                backend_argv=[*extract_argv, "&&", *rectify_argv, "&&", *colmap_argv, "--camera-params", "<from_rectified_intrinsics>"],
                passthrough=args.passthrough,
            )
            return 0
        _require_empty_or_overwrite(output_dir, args.overwrite)
        _invoke_module("data_preparation.rosbag_to_colmap.extract_rosbag_images", extract_argv)
        _invoke_module("data_preparation.rectification.fishpoly_to_pinhole", rectify_argv)
        colmap_argv.extend(["--camera-params", _pinhole_camera_params(rectified_scene_dir)])
        _invoke_module(
            "data_preparation.video2colmap.preprocess_video_to_colmap",
            _append_passthrough(colmap_argv, args.passthrough),
        )
        outputs = {
            "scene_dir": output_dir,
            "staging_raw_scene": raw_scene_dir,
            "staging_rectified_scene": rectified_scene_dir,
        }
        backend_args = {
            "source": "rosbag-sfm",
            "bag_dir": layout.bag_dir(args.scene),
            "calibration": layout.calibration_file(),
            "preset": args.preset,
            "overwrite": args.overwrite,
            "uses_odometry": False,
            "uses_lidar": False,
        }
    report = write_workflow_report(
        layout=layout,
        scene=args.scene,
        task="prepare",
        command="prepare",
        preset=args.preset,
        outputs=outputs,
        backend_args=backend_args,
    )
    print(f"[INFO] prepare report={report}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    layout = _layout_from_args(args)
    preset = get_preset(args.preset)
    output_dir = layout.validation_task_dir(args.scene, "projection")
    scene_dir = layout.lidar_scene_dir_for_preset(args.scene, args.preset)
    backend_argv = [
        "--scene-dir",
        str(scene_dir),
        "--output-dir",
        str(output_dir),
        "--report-path",
        str(output_dir / "report.json"),
        "--max-overlay-points",
        str(preset.max_overlay_points),
    ]
    if preset.projection_frames:
        backend_argv.append("--frames")
        backend_argv.extend(preset.projection_frames)
    _invoke_module(
        "data_preparation.data_quality.projection_overlay",
        _append_passthrough(backend_argv, args.passthrough),
    )
    report = write_workflow_report(
        layout=layout,
        scene=args.scene,
        task="projection",
        command="validate",
        preset=args.preset,
        outputs={"output_dir": output_dir, "report": output_dir / "report.json"},
        backend_args={"scene_dir": scene_dir, "preset": args.preset},
    )
    print(f"[INFO] validate report={report}")
    return 0


def cmd_colorize(args: argparse.Namespace) -> int:
    layout = _layout_from_args(args)
    preset = get_preset(args.preset)
    preset_scene_dir = layout.lidar_scene_dir_for_preset(args.scene, args.preset)
    scene_dir = preset_scene_dir if preset_scene_dir.exists() else layout.lidar_scene_dir(args.scene)
    output_dir = layout.validation_task_dir(args.scene, "colorize")
    output_ply = scene_dir / "lidar" / "global_map_colorized.ply"
    if args.preset == "smoke":
        output_ply = output_dir / "global_map_colorized_preview.ply"
    report_path = output_dir / "report.json"

    backend_argv = [
        "--scene-dir",
        str(scene_dir),
        "--output-ply",
        str(output_ply),
        "--report-path",
        str(report_path),
    ]
    if preset.projection_frames:
        backend_argv.append("--frames")
        backend_argv.extend(preset.projection_frames)
    if preset.colorize_sample_points > 0:
        backend_argv.extend(["--sample-points", str(preset.colorize_sample_points)])
    _invoke_module(
        "data_preparation.data_quality.colorize_lidar_map",
        _append_passthrough(backend_argv, args.passthrough),
    )
    report = write_workflow_report(
        layout=layout,
        scene=args.scene,
        task="colorize",
        command="colorize",
        preset=args.preset,
        outputs={"output_dir": output_dir, "output_ply": output_ply, "report": report_path},
        backend_args={
            "scene_dir": scene_dir,
            "sample_points": preset.colorize_sample_points,
            "frames": ",".join(preset.projection_frames or []),
            "preset": args.preset,
        },
    )
    print(f"[INFO] colorize report={report}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    layout = _layout_from_args(args)
    preset = get_preset(args.preset)
    rectified_scene_dir = layout.rectified_lidar_scene_dir(args.scene)
    use_rectified = bool(args.rectified or rectified_scene_dir.exists())
    output_dir = layout.colmap_compat_scene_dir(
        args.scene,
        suffix="pinhole_rectified_slam_compat" if use_rectified else "slam_compat",
    )
    scene_dir = rectified_scene_dir if use_rectified else layout.lidar_scene_dir_for_preset(args.scene, args.preset)
    colorized = scene_dir / "lidar" / "global_map_colorized.ply"
    points_ply = colorized if colorized.exists() else scene_dir / "lidar" / "global_map.ply"
    backend_argv = [
        "--scene-dir",
        str(scene_dir),
        "--output-dir",
        str(output_dir),
        "--points-ply",
        str(points_ply),
        "--max-points",
        str(preset.export_max_points),
    ]
    _invoke_module(
        "data_preparation.slam_to_colmap.main",
        _append_passthrough(backend_argv, args.passthrough),
    )
    report = write_workflow_report(
        layout=layout,
        scene=args.scene,
        task="export_colmap",
        command="export",
        preset=args.preset,
        outputs={"scene_dir": output_dir, "backend_report": output_dir / "slam_to_colmap_report.json"},
        backend_args={
            "format": args.format,
            "points_ply": points_ply,
            "preset": args.preset,
            "source_scene_dir": scene_dir,
            "rectified": use_rectified,
        },
    )
    print(f"[INFO] export report={report}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    args.passthrough = []
    if args.with_inspect:
        cmd_inspect(args)
    cmd_prepare(args)
    cmd_validate(args)
    return 0


WORKFLOWS: Dict[str, Callable[[argparse.Namespace], int]] = {
    "inspect": cmd_inspect,
    "prepare": cmd_prepare,
    "validate": cmd_validate,
    "colorize": cmd_colorize,
    "export": cmd_export,
    "run": cmd_run,
}


def run_workflow(command: str, argv: Optional[List[str]] = None) -> int:
    parser = build_parser(command)
    args = parser.parse_args(argv)
    try:
        return WORKFLOWS[command](args)
    except FileExistsError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
