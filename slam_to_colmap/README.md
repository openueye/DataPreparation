# SLAM to COLMAP-compatible

This tool converts a processed `images + poses.csv + camera.json + lidar/*.ply`
scene into a COLMAP-compatible text model:

```text
output/
├── images -> source/images
└── sparse/0/
    ├── cameras.txt
    ├── images.txt
    └── points3D.txt
```

For high-quality 3DGS training, the input images must already be rectified to
an undistorted `PINHOLE` or `SIMPLE_PINHOLE` model. The converter now refuses to
export non-pinhole camera models by default, because writing raw FishPoly images
as COLMAP `PINHOLE` creates a geometric mismatch that usually appears as blurry
renderings.

Legacy smoke-test export is still available with:

```bash
python -m data_preparation slam-to-colmap \
  --scene-dir /path/to/slam_scene \
  --output-dir /path/to/colmap_compat_scene \
  --allow-pinhole-approximation
```

The output is intended for loader compatibility and smoke tests when this legacy
flag is used. It is not a pure RGB-only COLMAP/SfM baseline because poses come
from SLAM and sparse points come from LiDAR.
