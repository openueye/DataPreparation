from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from data_preparation.shared.io import write_json
from data_preparation.shared.layout import DataPrepLayout


REPORT_NAME = "workflow_report.json"
STANDARD_TASKS = {
    "inspect": "inspect",
    "projection": "projection",
    "colorize": "colorize",
    "export_colmap": "export_colmap",
}


def task_report_path(layout: DataPrepLayout, scene: str, task: str) -> Path:
    return layout.validation_task_dir(scene, task) / REPORT_NAME


def standard_output_paths(layout: DataPrepLayout, scene: str, task: str) -> Dict[str, Path]:
    """Return wrapper-level output locations for common workflow tasks."""

    task_dir = layout.validation_task_dir(scene, STANDARD_TASKS.get(task, task))
    paths = {
        "task_dir": task_dir,
        "workflow_report": task_dir / REPORT_NAME,
    }
    if task == "inspect":
        paths.update(
            {
                "inventory": task_dir / "bag_inventory.md",
                "feasibility": task_dir / "feasibility_report.md",
                "summary": task_dir / "inspection_summary.json",
            }
        )
    elif task in {"projection", "colorize"}:
        paths["report"] = task_dir / "report.json"
    elif task == "export_colmap":
        paths["report"] = task_dir / "report.json"
    return paths


def write_workflow_report(
    *,
    layout: DataPrepLayout,
    scene: str,
    task: str,
    command: str,
    preset: Optional[str],
    outputs: Dict[str, Any],
    backend_args: Dict[str, Any],
    scope: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a small wrapper-level envelope without changing backend schemas."""

    payload = {
        "schema": "data_preparation.workflow.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scene": scene,
        "task": task,
        "command": command,
        "preset": preset,
        "outputs": {key: str(value) for key, value in outputs.items()},
        "backend_args": {key: str(value) for key, value in backend_args.items()},
    }
    if scope is not None:
        payload["scope"] = {key: str(value) for key, value in scope.items()}
    path = task_report_path(layout, scene, task)
    write_json(path, payload)
    return path
