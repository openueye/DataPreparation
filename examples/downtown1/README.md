# Downtown1 Example Workflow

This example keeps the validated Downtown1 data semantics and output naming
conventions. Adjust paths only when intentionally producing a new scene variant.

Run commands from:

```bash
cd /home/haibo/Documents/Thesis/00_Baselines
```

## 1. Export SLAM-Compatible COLMAP Scene

```bash
conda run -n 3dgs_train python -m data_preparation slam \
  --scene Downtown1 \
  --rosbag-dir /home/haibo/Documents/Thesis/03_Datasets/001_rosbags/Downtown1 \
  --output-dir /home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_1M \
  --max-points 1000000 \
  -- --copy-images
```

## 2. Generate Sparse LiDAR Depth

Raw single-frame baseline:

```bash
conda run -n 3dgs_train python -m data_preparation depth-prior-project \
  --scene-dir /home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_1M \
  --prepared-scene-dir /home/haibo/Documents/Thesis/04_ProcessedData/rosbag_prepared/Downtown1_pure_headerstamp \
  --method raw-frame \
  --min-depth 1.0 \
  --max-depth 50.0
```

Local fused9 sparse input:

```bash
conda run -n 3dgs_train python -m data_preparation depth-prior-project \
  --scene-dir /home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_1M \
  --prepared-scene-dir /home/haibo/Documents/Thesis/04_ProcessedData/rosbag_prepared/Downtown1_pure_headerstamp \
  --method local-fused \
  --fusion-window 9 \
  --min-depth 1.0 \
  --max-depth 50.0
```

## 3. Completed Depth Post-Processing

After SPNet anchor-preserved completion has produced:

```text
04_ProcessedData/slam/downtown1_1M/depths_spnet_fused9_anchor_preserved
```

generate edge90 masks:

```bash
conda run -n 3dgs_train python -m data_preparation depth-prior-edge-masks \
  --scene-dir /home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_1M \
  --completed-depths-dir /home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_1M/depths_spnet_fused9_anchor_preserved \
  --anchor-depths-dir /home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_1M/depths_lidar_fused_9f \
  --percentiles 90 \
  --target-min 1.0 \
  --target-max 50.0
```

apply the mask:

```bash
conda run -n 3dgs_train python -m data_preparation depth-prior-apply-mask \
  --scene-dir /home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_1M \
  --input-depths-dir /home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_1M/depths_spnet_fused9_anchor_preserved \
  --input-masks-dir /home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_1M/masks_spnet_fused9_anchor_preserved_edge90 \
  --anchor-depths-dir /home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_1M/depths_lidar_fused_9f \
  --output-depths-dir /home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_1M/depths_spnet_fused9_anchor_preserved_edge90 \
  --output-report /home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_1M/depth_prior_report_spnet_fused9_anchor_preserved_edge90.json
```

generate confidence and source-label sidecars:

```bash
conda run -n 3dgs_train python -m data_preparation depth-prior-sidecars \
  --final-depths-dir /home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_1M/depths_spnet_fused9_anchor_preserved_edge90 \
  --completed-depths-dir /home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_1M/depths_spnet_fused9_anchor_preserved \
  --anchor-depths-dir /home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_1M/depths_lidar_fused_9f \
  --confidence-dir /home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_1M/depth_confidence_spnet_fused9_anchor_preserved_edge90 \
  --source-labels-dir /home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_1M/depth_source_labels_spnet_fused9_anchor_preserved_edge90 \
  --report-path /home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_1M/depth_sidecar_report_spnet_fused9_anchor_preserved_edge90.json
```

## 4. Export to LSG-SLAM Layout

```bash
conda run -n 3dgs_train python -m data_preparation lsgslam-export \
  --source-scene /home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_1M \
  --depths-dir /home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_1M/depths_spnet_fused9_anchor_preserved_edge90 \
  --output-dir /home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_1M_lsgslam_euroc
```
