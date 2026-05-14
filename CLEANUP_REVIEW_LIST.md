# data_preparation cleanup review list

This file is an audit list only. Nothing listed here has been deleted.

## Formal route scripts to keep

These are the three official route entry scripts:

```text
sfm/main.py
hybrid/main.py
slam/main.py
```

Shared helpers for the route entry scripts live in:

```text
shared/route_helpers.py
```

The top-level CLI now delegates official commands to those route scripts:

```text
python -m data_preparation sfm
python -m data_preparation hybrid
python -m data_preparation slam
```

## Route backends to keep for now

These files are still used by the formal routes:

```text
video2colmap/preprocess_video_to_colmap.py      # backend for sfm/main.py
hybrid/converter.py                             # backend for hybrid/main.py
slam/export_colmap.py                           # backend for slam/main.py
slam/converter.py                               # SLAM conversion implementation
```

Shared modules used by the formal route scripts/backends:

```text
shared/layout.py
shared/io.py
shared/camera_models.py
shared/colmap_io.py
shared/pointcloud.py
shared/poses.py
```

## SLAM reference rule

For SLAM rosbag unpacking and pure-headerstamp scene generation, use the
reference implementation as the rule:

```text
Thesis/06_Referrance/DataprePython/organize_odin_scene_headerstamp.py
```

For pure-headerstamp to COLMAP conversion behavior, keep parity with:

```text
Thesis/06_Referrance/DataprePython/export_pure_headerstamp_to_colmap.py
```

The formal `slam` route should consume the pure-headerstamp scene layout. It
should not grow a separate rosbag-unpack implementation unless explicitly
approved.

## Deletion / archive candidates

These files are outside the three formal routes. Review before deleting because
some are still reachable through hidden legacy commands.

```text
workflows.py
rosbag_to_3dgs/README.md
rosbag_to_3dgs/convert_rosbag_to_3dgs.py
rosbag_to_3dgs/db.py
rosbag_to_3dgs/inspect_rosbags_for_3dgs.py
rosbag_to_3dgs/messages.py
rosbag_to_3dgs/ros2_cdr.py
rosbag_to_3dgs/validate_extrinsic_projection.py
rosbag_to_colmap/__init__.py
rosbag_to_colmap/extract_rosbag_images.py
data_quality/__init__.py
data_quality/colorize_lidar_map.py
data_quality/projection_overlay.py
shared/calibration.py
shared/presets.py
shared/reports.py
shared/timing.py
```

If these are deleted, also remove or adjust the hidden legacy commands in
`cli.py`:

```text
rosbag-inspect
rosbag-extract-images
```

## Documentation candidates

These docs describe legacy folder-level usage. Review after the route scripts
settle:

```text
video2colmap/README.md
```

## Already approved for deletion

These were approved for removal after the route-specific migration:

```text
rectification/
slam_to_colmap/
```

## Generated files / should not be tracked

These are cleanup candidates independent of route logic:

```text
__pycache__/
*/__pycache__/
*.pyc
```

Recommended follow-up after approval:

```text
1. Remove tracked __pycache__/*.pyc from git.
2. Add .gitignore rules for __pycache__/ and *.pyc.
3. Remove or archive the approved legacy/debug files.
```
