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
  --output-dir /path/to/hybrid_sfm_lidar_scene
```

Hybrid output is still a COLMAP-compatible scene. The trainer consumes it with
`--data-format colmap`.

## Deprecated Route

The old `slam-to-colmap` / `*_slam_compat` route is disabled. It used SLAM
camera poses plus LiDAR points, so it did not isolate LiDAR initialization from
camera-pose alignment. Historical outputs may remain archived for audit, but
new experiments should use either pure SfM/COLMAP or `hybrid-sfm-lidar`.

## Useful Commands

```bash
python -m data_preparation
python -m data_preparation inspect --scene Downtown1
python -m data_preparation prepare --scene Downtown1 --source rosbag-sfm --preset full
python -m data_preparation hybrid-sfm-lidar --help
python -m data_preparation video2colmap --help
```
