# Repository Conventions

## Overview

This repository implements a 3-stage spatial reasoning pipeline for the NYU WIRELESS
Summer 2026 interview task (Track 2: Spatial Reasoning). The pipeline operates on AI
Habitat's HM3D photorealistic scenes and integrates YOLO-World, Depth Pro, and FastSAM.

## Environment

Two Python environments isolate conflicting dependency requirements:

| Environment | Python | Manager | Purpose |
|:------------|:-------|:--------|:--------|
| `habitat-sim` | 3.9 | conda | 3D rendering (habitat-sim with Magnum bindings) |
| `.venv` | 3.12 | uv | ML models (PyTorch 2.11, YOLO-World, Depth Pro) |

- **Conda env** handles habitat-sim's native C++ dependencies (numpy 1.x, Magnum, bullet physics)
- **UV venv** handles the ML stack (numpy 2.x compatibility with PyTorch 2.11)
- `run_pipeline.sh` routes stages to the correct environment automatically

## Code Style

- **Python**: Target Python 3.9 compatibility for scripts running in the conda environment
  - Use `typing.Optional`, `typing.List`, `typing.Dict` (not `str | None`, `list[X]`)
  - Convert Magnum types to numpy arrays with `_to_np()` helper before numpy operations
- **Scripts**: Each pipeline stage is a standalone executable (`scripts/*.py`)
  - All accept `--input`, `--output`, and model-specific `--confidence`/`--model` flags
  - Output is always JSON (structured data) or JPEG (visual data)
- **No silent fallbacks**: Dependencies are explicit. Missing models or data produce clear
  error messages with installation instructions.

## Project Structure

```
scripts/           Pipeline stage executables (collect → detect → analyze)
docs/              Documentation site (GitHub Pages), architecture diagram, assets
data/              HM3D scene data (GLB + navmesh) — .gitignored
output/            Pipeline results (frames, detections, analysis) — .gitignored
```

## HM3D Data

The HM3D minival split requires a Matterport API token. Download instructions are in
`README.md`. The scene `00800-TEEsavR23oF` is the primary target per the task spec.

## Model Checkpoints

All `.pt` files are `.gitignored`. Models are downloaded automatically on first run:
- `yolov8s-worldv2.pt` (24.7 MB) — automatic via ultralytics
- `FastSAM-s.pt` (22.7 MB) — automatic via ultralytics
- `depth_pro.pt` (1.8 GB) — manual download via `get_pretrained_models.sh`

## Documentation

- **README.md**: Primary project documentation with embedded architecture diagram,
  pipeline demo composite, trajectory animation, real data tables, and setup instructions
- **docs/index.html**: Interactive scroll-driven GitHub Pages site with sidebar navigation,
  live data tables, and responsive dark theme
- **docs/architecture.html**: Standalone dark-themed SVG pipeline architecture diagram
- **AGENTS.md**: This file — repository conventions for contributors and AI assistants

## Verification

The pipeline has been verified end-to-end on:
1. **simple_room.glb** (habitat-sim test scene) — confirms rendering and data flow
2. **Real indoor photograph** — confirms YOLO-World + Depth Pro + FastSAM integration
3. **HM3D 00800-TEEsavR23oF** — full pipeline with 26 frames, 340 detections, 31 classes

To verify after changes: `bash run_pipeline.sh` and confirm all three stages produce output.
