# data_preparation

`data_preparation` prepares COLMAP-compatible scenes for `3DGS_baseline01`.
Training, checkpoints, viewers, and model code live in the training repository.

## Supported Routes

Pure visual COLMAP/SfM scene:

```bash
python -m data_preparation prepare \
  --scene Downtown1 \
  --source rosbag-sfm \
  --preset full
```

Video-to-COLMAP scene:

```bash
python -m data_preparation prepare \
  --scene MyScene \
  --source video \
  --video-path /path/to/video.mp4
```

Hybrid SfM-camera / LiDAR-seed scene:

```bash
python -m data_preparation hybrid-sfm-lidar \
  --scene-dir /path/to/filtered_slam_scene \
  --sfm-scene-dir /path/to/SFM_colmap_scene \
  --scene-name Downtown1_sparse322k
```

Hybrid output is still a COLMAP-compatible scene. The trainer consumes it with
`--data-format colmap`.

Default hybrid outputs should live under:

```text
04_ProcessedData/010_scenes_colmap/hybrid_sfm_lidar/<scene_variant>
```

The active processed-data categories are:

```text
sfm               # pure SfM/COLMAP
slam_compat       # deprecated SLAM camera + LiDAR points comparison
hybrid_sfm_lidar  # SfM cameras + transformed LiDAR seed
```

## Deprecated Route

The old `slam-to-colmap` / `*_slam_compat` route is disabled. It used SLAM
camera poses plus LiDAR points, so it did not isolate LiDAR initialization from
camera-pose alignment. The data can remain active as `slam_compat` for
comparison, but new LiDAR initialization experiments should use
`hybrid-sfm-lidar`.

## Useful Commands

```bash
python -m data_preparation
python -m data_preparation inspect --scene Downtown1
python -m data_preparation prepare --scene Downtown1 --source rosbag-sfm --preset full
python -m data_preparation hybrid-sfm-lidar --help
python -m data_preparation video2colmap --help
```
