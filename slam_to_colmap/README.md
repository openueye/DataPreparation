# SLAM and Hybrid COLMAP Exports

This package contains the two LiDAR-related formal routes:

```text
python -m data_preparation slam
python -m data_preparation hybrid
```

## slam

`slam` does not run SfM. It converts a pure-headerstamp SLAM/reference export
directly into a COLMAP text model:

```text
images_rectified/                  -> output images/
poses/camera_poses.csv             -> sparse/0/images.txt
calib/camera_rectified.json        -> sparse/0/cameras.txt
lidar/global_map_slam_odom.ply     -> sparse/0/points3D.txt
```

The pose CSV is expected to contain `T_odom_from_camera_*` matrix fields. The
output world frame follows the SLAM odom frame.

Use:

```bash
python -m data_preparation slam \
  --scene Ferrari1 \
  --input-dir /path/to/Ferrari1_pure_headerstamp
```

Default output:

```text
04_ProcessedData/slam_compat/<scene>
```

## hybrid

`hybrid` keeps the SfM/COLMAP cameras and image poses, then replaces the
initial point cloud with SLAM/LiDAR points:

```text
SfM images/ and sparse/0/cameras.bin/images.bin -> kept unchanged
SLAM poses/camera_poses.csv                     -> used only for alignment
SLAM lidar/global_map_slam_odom.ply             -> transformed into points3D.txt
```

The alignment is pose based: same-name SfM and SLAM camera centers are matched,
a similarity transform is estimated, and the colored SLAM point cloud is written
in the SfM coordinate frame.

Use:

```bash
python -m data_preparation hybrid \
  --scene Ferrari1 \
  --scene-dir /path/to/Ferrari1_pure_headerstamp \
  --sfm-scene-dir /home/haibo/Documents/Thesis/04_ProcessedData/sfm/Ferrari1 \
  --points-ply /path/to/Ferrari1_pure_headerstamp/lidar/global_map_slam_odom.ply
```

Default output:

```text
04_ProcessedData/hybrid_sfm_lidar/<scene>
```

Both outputs are trained with `3DGS_baseline01 --data-format colmap`.

Legacy command names such as `slam-to-colmap` and `hybrid-sfm-lidar` remain
callable for old scripts, but the formal CLI names are `slam` and `hybrid`.
