# Hybrid SfM-Camera / LiDAR-Seed Export

The supported LiDAR path builds a standard COLMAP-compatible scene for the
trainer:

```text
cameras.bin, images.bin, images/ -> SfM/COLMAP scene
points3D.txt                    -> SLAM/LiDAR points transformed into SfM coordinates
```

Use:

```bash
python -m data_preparation hybrid-sfm-lidar \
  --scene-dir /path/to/filtered_slam_scene \
  --sfm-scene-dir /path/to/SFM_colmap_scene \
  --scene-name Downtown1_sparse322k
```

When `--output-dir` is omitted, the output is written to:

```text
04_ProcessedData/010_scenes_colmap/hybrid_sfm_lidar/<scene-name>
```

The output is trained with `3DGS_baseline01 --data-format colmap`.

The old `slam-to-colmap` export is disabled. That path wrote SLAM camera poses
and LiDAR points into a COLMAP text layout (`*_slam_compat`), which is not a
clean LiDAR-initialization ablation because camera-pose alignment and point
initialization changed at the same time.
