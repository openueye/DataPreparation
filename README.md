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

## Usage

### SFM

```bash
python -m data_preparation sfm \
  --scene Ferrari1 \
  --image-dir /path/to/Ferrari1_pure_headerstamp/images_rectified \
  -- --camera-model PINHOLE --matcher sequential --sift-gpu
```

### Hybrid

```bash
python -m data_preparation hybrid \
  --scene Ferrari1 \
  --scene-dir /path/to/Ferrari1_pure_headerstamp \
  --sfm-scene-dir /home/haibo/Documents/Thesis/04_ProcessedData/sfm/Ferrari1 \
  --points-ply /path/to/Ferrari1_pure_headerstamp/lidar/global_map_slam_odom.ply \
  --max-points 3000000
```

### SLAM

```bash
python -m data_preparation slam \
  --scene Ferrari1 \
  --input-dir /path/to/Ferrari1_pure_headerstamp \
  --max-points 3000000 \
  -- --copy-images
```


python -m data_preparation slam \
  --scene Downtown1 \
  --rosbag-dir /Thesis/03_Datasets/001_rosbags/Downtown1
  --max-points 3000000 \
  --output-dir /path/to/downtown1_3M \
  -- --copy-images
  
## Training

All three routes write COLMAP-compatible scenes. The trainer consumes them with:

```bash
python train.py \
  --mode train \
  --data-format colmap \
  --data-dir /path/to/output_scene
```
