# data_preparation docs

This directory contains repository-local documentation for `data_preparation` routes and traceability records.

## Downtown1 depth-prior route closure

- Closure report: [2026-06-08_downtown1_depth_prior_route_closure_and_traceability.md](2026-06-08_downtown1_depth_prior_route_closure_and_traceability.md)

The Downtown1 depth-prior route covers raw-frame LiDAR projection, local multi-frame LiDAR fusion, SPNet completion, edge-mask filtering, confidence sidecars, source-label sidecars, and artifact provenance. `00_Baselines/data_preparation` is the canonical owner of depth-map generation and depth-prior artifact preparation; `00_Baselines/02baseline` only consumes prepared artifacts during training.

Depth-prior quality itself is the primary evaluation basis: coverage, LiDAR anchor preservation, completion validity, mask validity, confidence correctness, source-label traceability, and reproducibility. Downstream 3DGS metrics are secondary reference information only.
