# Downtown1 Depth Prior Route Closure and Traceability

## 1. Scope

This is a closure and traceability report for the Downtown1 depth-prior preparation route. It is not a new experiment, and it does not regenerate depth maps, train 3DGS, or move historical artifacts.

The canonical owner for depth-map generation and depth-prior preparation is:

```text
00_Baselines/data_preparation
```

The primary evaluation basis is depth-prior quality itself, not downstream 3DGS rendering metrics. Downstream 3DGS metrics are retained only as consumer-side traceability and smoke/full-run reference information.

Primary source logs inspected:

- `08_Logs/train_logs/2026-06-05_downtown1_slam_1m_depth_source_ablation/2026-06-05_downtown1_slam_1m_depth_source_ablation.md`
- `08_Logs/analysis/2026-06-05_depth_prior_coverage_reliability_strategy.md`
- `08_Logs/implementation/2026-06-06_02baseline_reliability_aware_depth_phase1.md`
- `08_Logs/train_logs/2026-06-06_downtown1_slam_1m_reliability_aware_depth/2026-06-06_downtown1_slam_1m_reliability_aware_depth.md`
- `08_Logs/analysis/2026-06-08_downtown1_work_summary_0605_0608.md`

## 2. Executive summary

The route evolved from global point-cloud projection to timestamp-aligned raw single-frame LiDAR projection, then local multi-frame LiDAR fusion, then SPNet completion, and finally reliability-aware confidence-weighted supervision.

The current recommended version is:

```text
SPNet fused9 anchor-preserved + edge90 mask + confidence weighting
```

This recommendation is based on depth-prior quality: coverage, LiDAR anchor preservation, completion validity, mask validity, confidence sidecar correctness, source-label traceability, and artifact reproducibility.

Downstream 3DGS metrics are recorded only as reference information. They should not be used as the primary evidence for this route because the current training experiments may not yet fully verify that the network uses the depth priors as intended.

## 3. Repository ownership and boundary

`00_Baselines/data_preparation` should own depth generation, depth reports, masks, confidence maps, source labels, and artifact provenance for the Downtown1 depth-prior route.

`00_Baselines/02baseline` should own training-time consumption only. Its boundary is the prepared-artifact interface:

```text
--depths-dir
--depth-masks-dir
--depth-confidence-dir
```

Current sufficiency judgment after the follow-up migration: `00_Baselines/data_preparation` is sufficient as the canonical owner for D1 raw-frame projection, D2 local-fused projection, and D4 post-completion edge-mask / final-depth / confidence-sidecar preparation. SPNet model inference for D3 is still represented by the historical external script because it depends on the Gaussian-LIC SPNet checkout, checkpoint loading, and GPU/CPU model runtime; it remains indexed here as a legacy external generation step until it receives its own tested wrapper.

Canonical post-completion commands now owned by `data_preparation`:

- `python -m data_preparation depth-prior-edge-masks`
- `python -m data_preparation depth-prior-apply-mask`
- `python -m data_preparation depth-prior-sidecars`

## 4. Route evolution

### 4.1 D0 Global point cloud projection

- Motivation: early exploratory or diagnostic projection of a global SLAM point cloud back into image frames to test depth-prior feasibility.
- Implementation / code location: not verified inside `00_Baselines/data_preparation` as a current method. The historical evidence is the older `04_ProcessedData/slam/downtown1_3M/depth_prior_report.json`, which lacks the current `method` contract.
- Artifact path: `04_ProcessedData/slam/downtown1_3M/depths`
- Report path: `04_ProcessedData/slam/downtown1_3M/depth_prior_report.json`
- Depth-quality result: historical report mean valid ratio recorded in logs as `18.12%`; not directly comparable to the current synchronized raw-frame/local-fused generator.
- Limitation: global projection lacks current route traceability and per-frame visibility/occlusion reasoning.
- Final status: diagnostic only / archived. Do not use as final supervision evidence.

### 4.2 D1 Raw single-frame LiDAR projection

- Motivation: geometrically clean synchronized raw LiDAR projection into each image frame.
- Implementation / code location: `00_Baselines/data_preparation/depth_prior/project.py`, `--method raw-frame`, `export_raw_frame_depth_priors`.
- Input data source: `associations/frame_associations.csv`, `lidar/raw_frames/*.npz`, `calib/tf_chain.json`, COLMAP-compatible scene images/cameras.
- Artifact path: `04_ProcessedData/slam/downtown1_1M/depths`
- Report path: `04_ProcessedData/slam/downtown1_1M/depth_prior_report.json`
- Generation command: documented in `00_Baselines/data_preparation/README.md` as `python -m data_preparation depth-prior-project --method raw-frame ...`.
- Depth-quality result: report mean valid ratio `1.1047%`.
- LiDAR anchor preservation evidence: raw projected pixels are direct LiDAR observations; separate anchor preservation check is not applicable.
- Limitation: too sparse for the current 3DGS depth-supervision objective.
- Final status: sparse anchor baseline / frozen raw-frame baseline.

### 4.3 D2 Local multi-frame fusion

- Motivation: increase sparse LiDAR coverage while retaining raw LiDAR geometry by fusing nearby timestamp-aligned raw frames with odometry compensation.
- Implementation / code location: `00_Baselines/data_preparation/depth_prior/project.py`, `--method local-fused`, `export_local_fused_depth_priors`, `_project_local_fused_depth`.
- Input data source: prepared scene `poses/camera_poses.csv`, `calib/tf_chain.json`, `calib/camera_rectified.json`, `associations/frame_associations.csv`, and `lidar/raw_frames/*.npz`.
- Artifact paths:
  - `04_ProcessedData/slam/downtown1_1M/depths_lidar_fused_5f`
  - `04_ProcessedData/slam/downtown1_1M/depths_lidar_fused_9f`
  - `04_ProcessedData/slam/downtown1_1M/depths_lidar_fused_15f`
- Report paths:
  - `04_ProcessedData/slam/downtown1_1M/depth_prior_report_local_fused_5f.json`
  - `04_ProcessedData/slam/downtown1_1M/depth_prior_report_local_fused_9f.json`
  - `04_ProcessedData/slam/downtown1_1M/depth_prior_report_local_fused_15f.json`
- Generation command: `conda run -n 3dgs_train python -m data_preparation depth-prior-project --method local-fused --fusion-window 5|9|15 --min-depth 1.0 --max-depth 50.0`.
- Depth-quality result: fused5 mean valid ratio `5.3302%`; fused9 `9.2455%`; fused15 `14.5043%`.
- Reliability notes: collision/occlusion risk rises with window size. Logs record mean z-buffer collision ratios of about `3.51%` for fused5, `6.62%` for fused9, and `11.00%` for fused15.
- Final status: fused5 is a sparse-fusion ablation; fused9 is the current practical sparse input to completion; fused15 is a higher-coverage pseudo/reference variant with higher collision risk.

### 4.4 D3 SPNet fused9 anchor-preserved completion

- Motivation: convert the practical fused9 sparse LiDAR input into near-dense metric depth while preserving real LiDAR anchors exactly.
- Implementation / code location: external experimental script `05_Outputs/depth_prior_experiments/2026-06-06_spnet_fused9_anchor_preserved_1600x1296/generate_spnet_anchor_preserved.py`.
- Artifact paths:
  - `04_ProcessedData/slam/downtown1_1M/depths_spnet_fused9_anchor_preserved`
  - `04_ProcessedData/slam/downtown1_1M/masks_spnet_fused9_anchor_preserved`
- Report path: `04_ProcessedData/slam/downtown1_1M/depth_prior_report_spnet_fused9_anchor_preserved.json`
- Generation command: `conda run -n 3dgs_train python 05_Outputs/depth_prior_experiments/2026-06-06_spnet_fused9_anchor_preserved_1600x1296/generate_spnet_anchor_preserved.py --overwrite`.
- Depth-quality result: `668` depth files, `668` masks, output shape recorded in logs as `1296x1600`, report mean valid ratio `99.3163%`.
- LiDAR anchor preservation evidence: sampled LiDAR anchor max absolute error recorded as `0.0 m`; D4 report also records anchor max absolute error summary as all zeros.
- Limitation: completed pixels are weaker than LiDAR anchors and need masks/confidence.
- Final status: completed-depth candidate requiring reliability filtering.

### 4.5 D4 Edge90 confidence-weighted supervision

- Motivation: keep near-dense coverage while rejecting high-risk edge regions and preventing SPNet-completed pixels from being supervised with the same trust as real LiDAR anchors.
- Implementation / code location:
  - edge mask generation and final masked depth application: `00_Baselines/data_preparation/depth_prior/edge_masks.py`
  - confidence/source-label sidecars: `00_Baselines/data_preparation/depth_prior/sidecars.py`
  - CLI dispatch: `python -m data_preparation depth-prior-edge-masks`, `depth-prior-apply-mask`, and `depth-prior-sidecars`
- Artifact paths:
  - depth: `04_ProcessedData/slam/downtown1_1M/depths_spnet_fused9_anchor_preserved_edge90`
  - mask: `04_ProcessedData/slam/downtown1_1M/masks_spnet_fused9_anchor_preserved_edge90`
  - confidence: `04_ProcessedData/slam/downtown1_1M/depth_confidence_spnet_fused9_anchor_preserved_edge90`
  - source labels: `04_ProcessedData/slam/downtown1_1M/depth_source_labels_spnet_fused9_anchor_preserved_edge90`
- Report paths:
  - `04_ProcessedData/slam/downtown1_1M/depth_prior_report_spnet_fused9_anchor_preserved_edge90.json`
  - `04_ProcessedData/slam/downtown1_1M/depth_prior_report_spnet_fused9_anchor_preserved_edge90_mask.json`
  - `04_ProcessedData/slam/downtown1_1M/depth_sidecar_report_spnet_fused9_anchor_preserved_edge90.json`
- Depth-quality result: edge90 final valid ratio mean `90.3092%`; anchor ratio mean `9.2455%`; anchor max absolute error summary `0.0 m`.
- Confidence contract: LiDAR anchor `1.0`, SPNet completed kept by edge90 `0.2`, invalid/rejected `0.0`.
- Final status: current recommended candidate.

## 5. Depth-quality quantitative summary

### Sparse Depth Coverage

| Variant | Depth path | Report path | Mean valid ratio | Reliability note |
| --- | --- | --- | ---: | --- |
| raw 1f | `04_ProcessedData/slam/downtown1_1M/depths` | `04_ProcessedData/slam/downtown1_1M/depth_prior_report.json` | `1.1047%` | geometrically clean but too sparse |
| fused 5f | `04_ProcessedData/slam/downtown1_1M/depths_lidar_fused_5f` | `04_ProcessedData/slam/downtown1_1M/depth_prior_report_local_fused_5f.json` | `5.3302%` | logs record mean collision about `3.51%` |
| fused 9f | `04_ProcessedData/slam/downtown1_1M/depths_lidar_fused_9f` | `04_ProcessedData/slam/downtown1_1M/depth_prior_report_local_fused_9f.json` | `9.2455%` | practical sparse input; logs record mean collision about `6.62%` |
| fused 15f | `04_ProcessedData/slam/downtown1_1M/depths_lidar_fused_15f` | `04_ProcessedData/slam/downtown1_1M/depth_prior_report_local_fused_15f.json` | `14.5043%` | higher coverage but logs record mean collision about `11.00%` |

### SPNet Completion Quality

| Check | Value | Evidence |
| --- | --- | --- |
| completed depth files | `668` | `depth_prior_report_spnet_fused9_anchor_preserved.json` |
| mask files | `668` | `depth_prior_report_spnet_fused9_anchor_preserved.json` and source log |
| image shape | `1296x1600` | source log, full-resolution artifact note |
| mean valid ratio | `99.3163%` | `valid_ratio_summary.mean` |
| input sparse depth | `depths_lidar_fused_9f` | `input_depths_dir` |
| LiDAR anchor preservation | preserved by overwrite | source log and edge90 anchor error report |
| sampled LiDAR anchor max absolute error | `0.0 m` | source log |
| report JSON | `04_ProcessedData/slam/downtown1_1M/depth_prior_report_spnet_fused9_anchor_preserved.json` | inspected |

### Confidence / Sidecar Quality

| Check | Value | Evidence |
| --- | --- | --- |
| confidence directory | `04_ProcessedData/slam/downtown1_1M/depth_confidence_spnet_fused9_anchor_preserved_edge90` | sidecar report |
| source-label directory | `04_ProcessedData/slam/downtown1_1M/depth_source_labels_spnet_fused9_anchor_preserved_edge90` | sidecar report |
| sidecar report JSON | `04_ProcessedData/slam/downtown1_1M/depth_sidecar_report_spnet_fused9_anchor_preserved_edge90.json` | inspected |
| number of images | `668` | sidecar report |
| anchor ratio mean | `9.2455%` | `anchor_ratio_summary.mean` |
| completed kept ratio mean | `81.0636%` | `completion_ratio_summary.mean` |
| rejected ratio mean | `9.0071%` | `rejected_ratio_summary.mean` |
| mean positive weight | `0.2819` | `mean_positive_weight_summary.mean` |
| source-label contract | `0 invalid`, `1 LiDAR anchor`, `2 SPNet kept`, `4 completed rejected` | sidecar report |
| confidence contract | anchor `1.0`, completed `0.2`, invalid/rejected `0.0` | sidecar report |

### Downstream 3DGS reference metrics, not primary depth-prior evidence

These metrics are retained for traceability only. They are not used as the primary success criterion for this depth-prior route because the downstream training pipeline may not yet be fully verified to consume and exploit the depth priors as intended.

| Run | Depth prior | Final L1 | Final SSIM | Final PSNR | Valid ratio | Metric source |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| no depth | none | `0.034261` | `0.824929` | `24.2959` | n/a | `08_Logs/train_logs/2026-06-05_downtown1_slam_1m_depth_source_ablation/...md` |
| raw single-frame | `depths` | `0.034525` | `0.826038` | `24.2139` | `1.11%` | same source log |
| fused 5f | `depths_lidar_fused_5f` | `0.033956` | `0.826712` | `24.3455` | `5.30%` | same source log |
| fused 9f sparse | `depths_lidar_fused_9f` | `0.034598` | `0.826582` | `24.2537` | `0.091729` | `08_Logs/train_logs/2026-06-06_downtown1_slam_1m_reliability_aware_depth/...md` |
| edge90 equal | `depths_spnet_fused9_anchor_preserved_edge90` | `0.034176` | `0.826093` | `24.2880` | `0.903584` | same source log |
| edge90 weighted | `depths_spnet_fused9_anchor_preserved_edge90` + mask/confidence | `0.034049` | `0.826522` | `24.3184` | `0.903584` | same source log |

## 6. Current recommended depth-prior version

Current recommendation:

```text
SPNet fused9 anchor-preserved + edge90 mask + confidence weighting
```

This recommendation is based on depth-map quality and reliability, not primarily on downstream 3DGS metrics.

| Item | Path / value |
| --- | --- |
| depth directory | `04_ProcessedData/slam/downtown1_1M/depths_spnet_fused9_anchor_preserved_edge90` |
| mask directory | `04_ProcessedData/slam/downtown1_1M/masks_spnet_fused9_anchor_preserved_edge90` |
| confidence directory | `04_ProcessedData/slam/downtown1_1M/depth_confidence_spnet_fused9_anchor_preserved_edge90` |
| source-label directory | `04_ProcessedData/slam/downtown1_1M/depth_source_labels_spnet_fused9_anchor_preserved_edge90` |
| depth prior report JSON | `04_ProcessedData/slam/downtown1_1M/depth_prior_report_spnet_fused9_anchor_preserved_edge90.json` |
| mask report JSON | `04_ProcessedData/slam/downtown1_1M/depth_prior_report_spnet_fused9_anchor_preserved_edge90_mask.json` |
| sidecar report JSON | `04_ProcessedData/slam/downtown1_1M/depth_sidecar_report_spnet_fused9_anchor_preserved_edge90.json` |
| SPNet generation command | `conda run -n 3dgs_train python 05_Outputs/depth_prior_experiments/2026-06-06_spnet_fused9_anchor_preserved_1600x1296/generate_spnet_anchor_preserved.py --overwrite` |
| sidecar generation command | recorded in `08_Logs/implementation/2026-06-06_02baseline_reliability_aware_depth_phase1.md` |
| code path inside `data_preparation` | raw-frame and local-fused sparse input generation only: `depth_prior/project.py` |
| `data_preparation` git commit | `65084e5c24c62a48499d06fbf01293d42d931678` |
| `data_preparation` dirty state before this report | untracked `slam/lsgslam_adapter.py`, `tests/test_lsgslam_adapter.py` |

Confidence contract:

- LiDAR anchor = `1.0`
- SPNet completed kept by edge90 = `0.2`
- invalid/rejected = `0.0`

Depth-quality reasons for this recommendation:

- fused9 provides a practical sparse LiDAR input with better coverage than raw 1f and lower expected collision risk than larger-window fusion;
- SPNet completion increases depth valid ratio to near-dense coverage;
- LiDAR anchors are preserved;
- edge90 removes unreliable completed regions near strong image/depth edges;
- confidence weighting prevents SPNet-completed pixels from being treated as equally reliable as real LiDAR observations.

## 7. Archived / deprecated variants

| Variant | Status | Reason |
| --- | --- | --- |
| global projection | diagnostic / archived | older route, not current method contract, visibility/occlusion not verified |
| raw 1f | sparse anchor baseline | clean synchronized LiDAR projection, but only about `1.10%` mean valid ratio |
| fused 5f | useful sparse-fusion ablation | improves coverage to about `5.33%`, but not the current completion input |
| fused 9f | current sparse input to completion | practical coverage/collision tradeoff |
| fused 15f | reference / cautionary variant | higher coverage but higher collision/occlusion risk |
| SPNet anchor-preserved without edge/confidence | completed candidate | near dense, but completed pixels need reliability gating |
| edge90 equal-weight SPNet | downstream ablation | useful consumer reference, but not recommended over weighted confidence |
| edge95 mask | not verified as current recommendation | artifact/report exists, but route closure selects edge90 weighted |

## 8. Artifact registry

| Variant | Artifact path | Artifact type | Report path | Verified status | Depth-quality notes |
| --- | --- | --- | --- | --- | --- |
| raw 1f | `04_ProcessedData/slam/downtown1_1M/depths` | metric depth `.npy` | `depth_prior_report.json` | verified | mean valid ratio `1.1047%` |
| fused 5f | `04_ProcessedData/slam/downtown1_1M/depths_lidar_fused_5f` | metric depth `.npy` | `depth_prior_report_local_fused_5f.json` | verified | mean valid ratio `5.3302%` |
| fused 5f mask | `04_ProcessedData/slam/downtown1_1M/masks_lidar_fused_5f` | mask `.npy` | `depth_prior_report_local_fused_5f.json` | verified | generated with local-fused |
| fused 9f | `04_ProcessedData/slam/downtown1_1M/depths_lidar_fused_9f` | metric depth `.npy` | `depth_prior_report_local_fused_9f.json` | verified | mean valid ratio `9.2455%` |
| fused 9f mask | `04_ProcessedData/slam/downtown1_1M/masks_lidar_fused_9f` | mask `.npy` | `depth_prior_report_local_fused_9f.json` | verified | sparse input mask |
| fused 15f | `04_ProcessedData/slam/downtown1_1M/depths_lidar_fused_15f` | metric depth `.npy` | `depth_prior_report_local_fused_15f.json` | verified | mean valid ratio `14.5043%`, higher collision risk |
| fused 15f mask | `04_ProcessedData/slam/downtown1_1M/masks_lidar_fused_15f` | mask `.npy` | `depth_prior_report_local_fused_15f.json` | verified | reference mask |
| SPNet fused9 anchor-preserved | `04_ProcessedData/slam/downtown1_1M/depths_spnet_fused9_anchor_preserved` | completed metric depth `.npy` | `depth_prior_report_spnet_fused9_anchor_preserved.json` | verified | mean valid ratio `99.3163%` |
| SPNet fused9 anchor mask | `04_ProcessedData/slam/downtown1_1M/masks_spnet_fused9_anchor_preserved` | mask `.npy` | `depth_prior_report_spnet_fused9_anchor_preserved.json` | verified | `668` masks |
| SPNet edge90 depth | `04_ProcessedData/slam/downtown1_1M/depths_spnet_fused9_anchor_preserved_edge90` | final candidate depth `.npy` | `depth_prior_report_spnet_fused9_anchor_preserved_edge90.json` | verified | mean valid ratio `90.3092%` |
| SPNet edge90 mask | `04_ProcessedData/slam/downtown1_1M/masks_spnet_fused9_anchor_preserved_edge90` | final candidate mask `.npy` | `depth_prior_report_spnet_fused9_anchor_preserved_edge90_mask.json` | verified | edge90 filtered mask |
| SPNet edge95 mask | `04_ProcessedData/slam/downtown1_1M/masks_spnet_fused9_anchor_preserved_edge95` | alternative mask `.npy` | `depth_prior_report_spnet_fused9_anchor_preserved_edge95_mask.json` | verified artifact, not recommended | mean valid ratio `94.8127%`; not selected |
| edge90 confidence | `04_ProcessedData/slam/downtown1_1M/depth_confidence_spnet_fused9_anchor_preserved_edge90` | confidence sidecar `.npy` | `depth_sidecar_report_spnet_fused9_anchor_preserved_edge90.json` | verified | anchor `1.0`, completed `0.2` |
| edge90 source labels | `04_ProcessedData/slam/downtown1_1M/depth_source_labels_spnet_fused9_anchor_preserved_edge90` | source-label sidecar `.npy` | `depth_sidecar_report_spnet_fused9_anchor_preserved_edge90.json` | verified | labels `0/1/2/4` |
| SPNet overlays | `04_ProcessedData/slam/downtown1_1M/depth_prior_overlays_spnet_fused9_anchor_preserved` | visual inspection overlays | `depth_prior_report_spnet_fused9_anchor_preserved.json` | verified path | lightweight visual checks |
| historical 3M | `04_ProcessedData/slam/downtown1_3M/depths` | historical depth `.npy` | `04_ProcessedData/slam/downtown1_3M/depth_prior_report.json` | verified path, archived | older global projection route; not current evidence |
| SPNet pilot summaries | `05_Outputs/depth_prior_experiments/2026-06-05_completion_pilot/` | JSON summaries | multiple `spnet_*_640x512.json` | verified path | coverage/holdout pilot |

## 9. data_preparation code registry

| Code file | Inside `data_preparation` | Role | Relevant functions/classes/CLI | Status | Migration/indexing note |
| --- | --- | --- | --- | --- | --- |
| `00_Baselines/data_preparation/cli.py` | yes | route dispatcher | `depth-prior-project` command dispatch | production | keep |
| `00_Baselines/data_preparation/__main__.py` | yes | module entry point | `python -m data_preparation` | production | keep |
| `00_Baselines/data_preparation/depth_prior/project.py` | yes | raw-frame and local-fused depth generation, z-buffering, reports | `export_depth_priors`, `export_raw_frame_depth_priors`, `export_local_fused_depth_priors`, `_project_local_fused_depth` | production | keep and extend for future canonical wrappers |
| `00_Baselines/data_preparation/shared/organize_odin_scene.py` | yes | prepared ROS bag scene, timestamp associations, raw LiDAR frames, calibration chain | `export_pose_and_association_tables`, `build_tf_chain`, `export_raw_clouds` | production support | keep |
| `00_Baselines/data_preparation/shared/camera_models.py` | yes | camera projection helpers | `project_pinhole_k_like`, rectification helpers | support | keep |
| `00_Baselines/data_preparation/shared/poses.py` | yes | pose transform helpers | `matrix_from_pose_row`, `camera_from_world` | support | keep |
| `00_Baselines/data_preparation/shared/io.py` | yes | JSON/CSV/image path helpers | `load_json`, `write_json`, `load_csv_rows` | support | keep |
| `00_Baselines/data_preparation/tests/test_depth_prior_project.py` | yes | projection and report tests | raw-frame and local-fused tests | test | keep |
| `00_Baselines/data_preparation/depth_prior/edge_masks.py` | yes | edge-mask generation and mask application | `generate_edge_masks`, `apply_depth_mask`, `depth-prior-edge-masks`, `depth-prior-apply-mask` | production post-completion utility | canonical replacement for old experiment scripts |
| `00_Baselines/data_preparation/depth_prior/sidecars.py` | yes | confidence and source-label sidecar generation | `generate_sidecars`, `depth-prior-sidecars` | production post-completion utility | canonical replacement for old `02baseline/tools` generator |
| `00_Baselines/data_preparation/tests/test_depth_prior_edge_masks.py` | yes | edge-mask and apply-mask regression tests | small `.npy` fixtures | test | keep |
| `00_Baselines/data_preparation/tests/test_depth_prior_sidecars.py` | yes | confidence/source-label regression tests | small `.npy` fixtures | test | keep |
| `00_Baselines/data_preparation/tests/test_cli_depth_prior_commands.py` | yes | CLI command registry test | `DEPTH_PRIOR_COMMANDS` | test | keep |
| `05_Outputs/depth_prior_experiments/2026-06-06_spnet_fused9_anchor_preserved_1600x1296/generate_spnet_anchor_preserved.py` | no | SPNet completion and anchor preservation | script CLI | experimental / external | migrate or wrap under `data_preparation` later |
| `05_Outputs/depth_prior_experiments/2026-06-06_spnet_fused9_anchor_preserved_1600x1296/generate_edge_gated_masks.py` | no | edge mask generation | script CLI | migrated / obsolete | replaced by `depth-prior-edge-masks`; removed after implementation log |
| `05_Outputs/depth_prior_experiments/2026-06-06_spnet_fused9_anchor_preserved_1600x1296/apply_mask_to_depths.py` | no | apply mask to completed depths | script CLI | migrated / obsolete | replaced by `depth-prior-apply-mask`; removed after implementation log |
| `00_Baselines/02baseline/tools/generate_depth_confidence_sidecars.py` | no | confidence and source-label sidecar generation | script CLI | migrated / obsolete | replaced by `depth-prior-sidecars`; removed after implementation log |

Global point-cloud projection code for D0 was not verified as an active `data_preparation` method. Treat the historical 3M report as archived evidence only.

## 10. 02baseline consumer interface

This section documents only consumption. Depth generation and depth-prior artifact preparation belong to `00_Baselines/data_preparation`.

| File | Interface | Role | Status |
| --- | --- | --- | --- |
| `00_Baselines/02baseline/cli/runtime_args.py` | `--depths-dir`, `--depth-masks-dir`, `--depth-confidence-dir` | parse consumer-side paths | consumer boundary |
| `00_Baselines/02baseline/configs/base.py` | depth prior config fields | stores consumer configuration | consumer boundary |
| `00_Baselines/02baseline/scene_data/camera.py` | `Camera.get_depth_prior()` | loads target depth, explicit mask, and confidence weight | consumer boundary |
| `00_Baselines/02baseline/losses/depth_loss.py` | weighted depth loss | computes weighted `inverse_l1` depth loss | consumer boundary |
| `00_Baselines/02baseline/trainer/gaussian_trainer.py` | reliability metrics | logs depth valid/weight metrics | consumer boundary |
| `00_Baselines/02baseline/tests/test_depth_prior.py` | unit tests | verifies loading, masks, confidence, and depth loss behavior | consumer test |
| `00_Baselines/02baseline/tests/test_paths_layout.py` | unit tests | verifies path layout behavior | consumer test |

Downstream `02baseline` / 3DGS metrics are secondary consumer references only, not the primary basis for closing the depth-prior route.

## 11. Downstream consumer reference

These downstream metrics are not the primary evaluation basis of the depth-prior construction route. They are retained only to show which depth-prior artifacts were consumed by `02baseline` and to preserve experiment provenance.

| Output path | Depth prior directory | Mask directory | Confidence directory | Verification type | Final PSNR / SSIM / L1 | Metric source log |
| --- | --- | --- | --- | --- | --- | --- |
| `05_Outputs/Downtown1/slam/0605_1M_clean_nodepth_nodensify_nosizeprune` | none active | none | none | full 30k no-depth reference | `24.2959 / 0.824929 / 0.034261` | 2026-06-05 depth source log |
| `05_Outputs/Downtown1/slam/0605_1M_clean_raw_single_depth_nodensify_nosizeprune` | `04_ProcessedData/slam/downtown1_1M/depths` | none | none | full 30k | `24.2139 / 0.826038 / 0.034525` | 2026-06-05 depth source log |
| `05_Outputs/Downtown1/slam/0605_1M_clean_fused5_depth_nodensify_nosizeprune` | `04_ProcessedData/slam/downtown1_1M/depths_lidar_fused_5f` | none | none | full 30k | `24.3455 / 0.826712 / 0.033956` | 2026-06-05 depth source log |
| `05_Outputs/Downtown1/slam/0605_1M_clean_fused9_depth_nodensify_nosizeprune` | `04_ProcessedData/slam/downtown1_1M/depths_lidar_fused_9f` | none | none | full 30k | `24.2537 / 0.826582 / 0.034598` | 2026-06-06 reliability-aware log |
| `05_Outputs/Downtown1/slam/0606_1M_clean_spnet_fused9_edge90_depth_nodensify_nosizeprune` | `04_ProcessedData/slam/downtown1_1M/depths_spnet_fused9_anchor_preserved_edge90` | none | none | full 30k equal-weight consumer run | `24.2880 / 0.826093 / 0.034176` | 2026-06-06 reliability-aware log |
| `05_Outputs/Downtown1/slam/0606_1M_clean_spnet_fused9_edge90_weighted_confidence_depth_nodensify_nosizeprune` | `04_ProcessedData/slam/downtown1_1M/depths_spnet_fused9_anchor_preserved_edge90` | `04_ProcessedData/slam/downtown1_1M/masks_spnet_fused9_anchor_preserved_edge90` | `04_ProcessedData/slam/downtown1_1M/depth_confidence_spnet_fused9_anchor_preserved_edge90` | full 30k weighted consumer run | `24.3184 / 0.826522 / 0.034049` | 2026-06-06 reliability-aware log |
| `05_Outputs/Downtown1/slam/0606_1M_clean_spnet_fused9_edge90_weighted_confidence_smoke200` | same as weighted edge90 | same as weighted edge90 | same as weighted edge90 | 200-iteration smoke loading/runtime check | no validation by design | 2026-06-06 implementation log |

## 12. Git version and dirty state

Main Thesis repository:

```text
git -C /home/haibo/Documents/Thesis branch --show-current
fatal: not a git repository (or any of the parent directories): .git
```

`/home/haibo/Documents/Thesis` contains a read-only `.git` path in this environment, but `git -C /home/haibo/Documents/Thesis ...` does not resolve it as a usable work tree. Main repo commit hash is therefore not available from this workspace state.

`00_Baselines/data_preparation` nested repository:

```text
branch: main
HEAD: 65084e5c24c62a48499d06fbf01293d42d931678
status before this report:
?? slam/lsgslam_adapter.py
?? tests/test_lsgslam_adapter.py
diff --stat before this report: empty
```

`00_Baselines/02baseline` nested repository:

```text
branch: main
HEAD: a8ee0b9741c6bcb7447bcebe196e395dde894bc8
status --short: empty
diff --stat: empty
diff --name-only: empty
```

## 13. Remaining risks

- Completed SPNet pixels are weaker than LiDAR anchors.
- Fused-window expansion may increase collision and occlusion errors.
- Depth-prior gain should be evaluated primarily through depth quality, not downstream 3DGS metrics for now.
- Downstream 3DGS consumption is not yet the main proof of the depth-prior route.
- Current result is Downtown1 1M specific and should be validated with further depth-quality ablations.
- Global projection should not be reused as final supervision unless visibility reasoning is added.
- `data_preparation` still needs a future tested wrapper for SPNet model inference if D3 is to be fully production-owned rather than externally indexed.

## 14. Next minimal depth-prior ablations

- completed confidence `0.1`;
- completed confidence `0.05`;
- anchor-only supervision + completed-depth initialization;
- edge80 / edge90 / edge95;
- fused5-completed versus fused9-completed if time permits;
- direct depth-quality validation before relying on downstream 3DGS metrics.

## 15. Final closure statement

This route is now closed as a traceable depth-prior construction module under:

```text
00_Baselines/data_preparation
```

The recommended candidate is:

```text
Reliability-aware completed LiDAR depth prior:
SPNet fused9 anchor-preserved + edge90 + confidence weighting.
```

This recommendation is based on depth-prior quality criteria: coverage, anchor preservation, completion validity, mask validity, confidence sidecar design, source-label traceability, and artifact reproducibility.

Downstream 3DGS metrics are retained only as reference information and should not be treated as the main evidence for this closure.

Further work should refine confidence modeling, edge filtering, and possible initialization usage. The route should not be restarted from global point cloud projection unless explicit per-frame visibility and occlusion reasoning is added.
