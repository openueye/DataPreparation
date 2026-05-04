# rosbag_to_3dgs

This package is retained for historical inspection code and archived
SLAM/LiDAR experiment provenance. Its conversion and projection-validation
commands are no longer exposed through `python -m data_preparation`.

Current training experiments should use:

```text
rosbag-sfm / video -> COLMAP scene -> 3DGS_baseline01 --data-format colmap
```

For LiDAR-seed ablations, use the retained hybrid exporter:

```bash
python -m data_preparation hybrid-sfm-lidar --help
```

The old direct `slam_lidar` training handoff is deprecated and unsupported.
