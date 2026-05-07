# Habitat Spatial Reasoning Pipeline

End-to-end egocentric perception on HM3D photorealistic scenes — trajectory collection, open-vocabulary object detection, monocular depth estimation, and per-object spatial analysis.

*NYU WIRELESS · Prof. Sundeep Rangan · Summer 2026 · Track 2: Spatial Reasoning*

---

<p align="center">
  <img src="docs/assets/architecture.svg" alt="Pipeline architecture" width="100%">
</p>

**Figure 1:** Three-stage pipeline. HM3D scene → trajectory collection (habitat-sim) → object detection (YOLO-World v2) → depth estimation + segmentation (Depth Pro + FastSAM). Stages communicate via JPEG/JSON interfaces.

---

## Results

All metrics measured on HM3D scene `00800-TEEsavR23oF` — a 398 m² furnished residential space (2 floors, 7 rooms). Hardware: Apple M1 Max.

| | |
|---|---|
| Trajectory frames | 26 |
| Object detections | 340 |
| Unique classes | 31 |
| Depth range | 1.12 – 16.05 m |
| Mean detections/frame | 13.1 |
| Detection confidence threshold | 0.15 |

<p align="center">
  <img src="docs/assets/pipeline_demo.jpg" alt="Pipeline demo" width="100%">
</p>

**Figure 2:** Frame 0001. Left: original RGB. Center: Depth Pro metric depth (Inferno, 1.7–12.7 m). Right: YOLO-World v2 — 23 objects, 14 classes.

### Per-object depth (Frame 0000)

| Object | Confidence | Mean depth | Std | Range | Mask pixels |
|---|---|---|---|---|---|
| couch | 0.86 | 3.11 m | 0.41 | 2.28 – 4.64 m | 36,527 |
| ceiling | 0.50 | 3.92 m | 0.56 | 2.35 – 6.11 m | 67,403 |
| sofa | 0.44 | 4.59 m | 0.36 | 3.54 – 5.69 m | 2,676 |
| chair | 0.33 | 6.79 m | 0.68 | 5.43 – 8.56 m | 1,051 |
| floor | 0.35 | 4.39 m | 2.10 | 2.95 – 14.66 m | 14,646 |

### Detected classes (31)

`cabinet` · `ceiling` · `wall` · `couch` · `pillow` · `floor` · `window` · `chair` · `lamp` · `mirror` · `rug` · `door` · `sofa` · `table` · `curtain` · `desk` · `oven` · `picture` · `plant` · `refrigerator` · `sink` · `television` · `clock` · `book` · `bag` · `bottle` · `bowl` · `bed` · `person` · `tv` · `vase`

### Depth ranges by frame

| Frame | Min | Max | Span | Notable |
|---|---|---|---|---|
| 0000 | 2.28 m | 16.05 m | 13.77 m | couch at 3.11 m, chair at 6.79 m |
| 0001 | 1.65 m | 12.67 m | 11.02 m | refrigerator at 5.39 m |
| 0002 | 2.61 m | 4.12 m | 1.51 m | narrow corridor — consistent depth |
| 0003 | 1.86 m | 8.48 m | 6.62 m | bedroom with open doorway |
| 0004 | 1.12 m | 9.11 m | 7.99 m | close to cabinet, open sightline |

---

## Trajectory

<p align="center">
  <img src="docs/assets/trajectory_walkthrough.gif" width="480" alt="Trajectory walkthrough">
</p>

**Figure 3:** 26-frame navmesh-constrained walkthrough. Agent at 1.5 m eye height, 90° HFOV, 640×480. Quaternion gaze toward next waypoint.

---

## Setup

Two isolated Python environments resolve dependency conflicts between habitat-sim's native C++ bindings and the ML stack:

| Environment | Python | Manager | Role |
|---|---|---|---|
| `habitat-sim` | 3.9 | conda | 3D rendering (habitat-sim 0.3.3, Magnum, numpy < 2) |
| `.venv` | 3.12 | uv | ML models (PyTorch 2.11, YOLO-World, Depth Pro) |

```bash
# Full pipeline
bash run_pipeline.sh

# Individual stages
conda run -n habitat-sim python scripts/collect_trajectory.py \
  --scene data/hm3d/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb --frames 25

source .venv/bin/activate
python scripts/detect_objects.py --confidence 0.15
python scripts/analyze_scene.py --max-frames 5
```

Full installation instructions in [`docs/index.html`](docs/index.html).

---

## Design

**Modular stages.** Each stage reads and writes JPEG/JSON — standard formats enabling independent debugging, model swapping, and distributed execution.

**Open-vocabulary detection.** Objects are specified by natural language prompts via CLIP text encoder. The vocabulary is a CLI argument — no retraining needed.

**Navmesh navigation.** Waypoints are constrained to walkable regions, producing realistic human-like paths rather than random floating-camera trajectories.

**Dual environments.** Isolating habitat-sim (numpy < 2, Magnum) from ML models (numpy 1.26, PyTorch 2.11) prevents dependency conflicts while keeping both ecosystems current.

---

## Limitations

| Constraint | Mitigation |
|---|---|
| Depth Pro runs at 5–10 s/frame on CPU | Batch overnight; MPS acceleration pending |
| FastSAM internal 640 px resize causes mask artifacts | Bilinear interpolation upgrade |
| No temporal object tracking across frames | Hungarian matching + Kalman filter |
| Textureless surfaces yield near-zero detections | Expected for geometric test scenes; HM3D is textured |
| Thin structures below detection threshold | SAM-family models struggle with sub-10 px features |
| Monocular depth ambiguity at boundaries | Denser trajectory improves multi-view consistency |

---

## References

[AI Habitat](https://aihabitat.org/) · [HM3D Dataset v0.2](https://aihabitat.org/datasets/hm3d/) · [YOLO-World (CVPR 2024)](https://github.com/AILab-CVC/YOLO-World) · [Depth Pro (Apple, 2024)](https://github.com/apple/ml-depth-pro) · [NanoSAM](https://github.com/NVIDIA-AI-IOT/nanosam) · [FastSAM (ICCV 2023)](https://github.com/CASIA-IVA-Lab/FastSAM)

---

*Interactive documentation: [asheshkaji.com/habitat](http://asheshkaji.com/habitat/)*
