# data_preparation

`data_preparation` prepares COLMAP-compatible scenes for `3DGS_baseline01`.
Training, checkpoints, viewers, and model code live in the training repository.

## Formal Routes

The public data-preparation surface is intentionally limited to three routes:

```text
sfm     rectified images -> COLMAP SfM scene
hybrid  SfM cameras/poses + aligned SLAM/LiDAR points -> COLMAP scene
slam    SLAM poses + SLAM/LiDAR points -> COLMAP text scene
```

The route entry scripts are collected under route-specific folders:

```text
sfm/main.py
hybrid/main.py
slam/main.py
```

Use the top-level command to see only these formal routes:

```bash
python -m data_preparation
```

Legacy/debug commands are still callable directly for reproducibility, but they
are hidden from the default summary.

## SFM

Input:

```text
images_rectified/
```

Processing:

```text
Run COLMAP only on the rectified images.
```

Output:

```text
04_ProcessedData/sfm/<scene>/
├── images/
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

Example:

```bash
python -m data_preparation sfm \
  --scene Ferrari1 \
  --image-dir /path/to/Ferrari1_pure_headerstamp/images_rectified \
  -- --camera-model PINHOLE --matcher sequential --sift-gpu
```

Pass `--output-dir` to override the default `04_ProcessedData/sfm/<scene>`
location.

## Hybrid

Input:

```text
SLAM/reference scene:
  images_rectified/
  poses/camera_poses.csv
  lidar/global_map_slam_odom.ply

SfM scene:
  images/
  sparse/0/cameras.bin
  sparse/0/images.bin
```

Processing:

```text
Keep SfM cameras, images, and poses.
Estimate a similarity transform from same-name SfM and SLAM camera centers.
Transform the colored SLAM/LiDAR point cloud into the SfM coordinate frame.
Write those points as sparse/0/points3D.txt.
```

Output:

```text
04_ProcessedData/hybrid_sfm_lidar/<scene>/
├── images -> SfM images
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    └── points3D.txt
```

Example:

```bash
python -m data_preparation hybrid \
  --scene Ferrari1 \
  --scene-dir /path/to/Ferrari1_pure_headerstamp \
  --sfm-scene-dir /home/haibo/Documents/Thesis/04_ProcessedData/sfm/Ferrari1 \
  --points-ply /path/to/Ferrari1_pure_headerstamp/lidar/global_map_slam_odom.ply \
  -- --max-points 3000000
```

Pass `--output-dir` to override the default
`04_ProcessedData/hybrid_sfm_lidar/<scene>` location.

## SLAM

Input:

```text
images_rectified/
poses/camera_poses.csv              # uses T_odom_from_camera_*
calib/camera_rectified.json
associations/frame_associations.csv
lidar/slam_frames_manifest.csv
lidar/global_map_slam_odom.ply
```

Processing:

```text
Do not run SfM.
Convert SLAM camera poses and the colored SLAM/LiDAR map into a COLMAP text model.
```

Output:

```text
04_ProcessedData/slam_compat/<scene>/
├── images/
└── sparse/0/
    ├── cameras.txt
    ├── images.txt
    └── points3D.txt
```

Example:

```bash
python -m data_preparation slam \
  --scene Ferrari1 \
  --input-dir /path/to/Ferrari1_pure_headerstamp \
  -- --copy-images
```

Pass `--output-dir` to override the default
`04_ProcessedData/slam_compat/<scene>` location.

## Training

All three routes write COLMAP-compatible scenes. The trainer consumes them with:

```bash
python train.py \
  --mode train \
  --data-format colmap \
  --data-dir /path/to/output_scene
```
